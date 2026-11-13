# SpaConAE development log

This file records how the model came to be: the decisions taken, the alternatives that were tried or only discussed, and the reasons. It holds the detail that fits neither the README (how to use the code) nor the project report (the final model and its results).

It was rebuilt from summaries of the design discussions held with ChatGPT during development, cross-checked against the notebooks, the report and the code. The order of the early entries is approximate: the discussions were not dated, and the implementation threads ran in parallel with the design ones.

## Contents

1. [Notation](#1-notation)
2. [Decision log](#2-decision-log)
3. [Design notes that still hold](#3-design-notes-that-still-hold)
4. [Ideas not pursued](#4-ideas-not-pursued)
5. [Evaluation checklist](#5-evaluation-checklist)
6. [Known issues and changes](#6-known-issues-and-changes)
7. [Literature map](#7-literature-map)

## 1. Notation

The discussions used `P`, `W` and `M` for the alignment matrix, in both orientations. This log uses one convention, the one of the code.

| Symbol | Shape | Meaning |
|---|---|---|
| $S$ | $n_{sc} \times g$ | scRNA-seq expression on the shared genes |
| $G$ | $n_{st} \times g$ | spatial expression on the shared genes |
| $Z_{sc}$, $Z_{st}$ | $n \times d$ | latent representations from the two encoders |
| $\tilde M$ | $n_{sc} \times n_{st}$ | alignment logits, the trainable parameter |
| $M$ | $n_{sc} \times n_{st}$ | row-wise softmax of $\tilde M$: each cell is a distribution over spots |
| $M^\top Z_{sc}$ | $n_{st} \times d$ | pseudo-spots: scRNA-seq latents projected onto the spots |
| $\tau$ | scalar | similarity threshold of the contrastive loss (`sim_threshold`) |

Status labels used below: **Adopted** (in the final model), **Discarded** (tried and abandoned), **Superseded** (replaced by a later decision), **Open** (not settled).

## 2. Decision log

### D1. Explicit factorisation or learned representation?

- **Problem.** The starting question was how to build a deep-learning model close to Tangram, which learns a constrained map so that $G \approx M^\top S$.
- **Options.** (a) An explicit NMF-like factorisation, possibly as an autoencoder with non-negative weights (encoder gives activations, decoder gives gene loadings); iNMF/LIGER-style shared and dataset-specific factors. (b) Latent representation learning with autoencoders, aligning the two modalities in latent space. (c) Generating the map from learned embeddings, $M_{ij} = \mathrm{softmax}_j(\mathrm{sim}(z_i, z_j))$, instead of treating it as a free parameter.
- **Decision.** (b): autoencoders with a shared latent space, with a Tangram-style map on top.
- **Reason.** Tangram is not NMF in the strict sense: $S$ is fixed and only the map is learned. A "deep-learning model integrating scRNA-seq and ST" and a "neural NMF" are different objectives; the goal here was integration and cell-to-spot mapping, not interpretable factors.
- **Status.** Adopted. Options (a) and (c) are in [section 4](#4-ideas-not-pursued).

### D2. Start from stAI and remove its task-specific heads

- **Problem.** A concrete architecture to start from.
- **Decision.** Use the stAI design: an MLP autoencoder for scRNA-seq, a graph-attention encoder with MLP decoder for spatial data, same latent size, and the cross-decoder cosine "consistency" loss. Drop stAI's supervised and task objectives (cell-type classifier, attention-based imputation, MMD, gene-graph loss) to obtain a task-independent, self-supervised model.
- **Details.** Training uses the intersection of the two gene sets; using the union would introduce modality-specific missing features that would need explicit modelling. The scRNA-seq encoder has one more hidden layer than stAI's to keep up with the spatial encoder.
- **Status.** Adopted.

### D3. Same latent size is not a shared latent space

- **Problem.** What makes the two latents comparable.
- **Reasoning.** Giving both encoders the same output dimension does not align them; the loss does. With paired observations a sample-wise loss (cosine, MSE, InfoNCE) is enough. Here the observations are unpaired, so sample-level equality is undefined and alignment must come from pseudo-pairing or from distribution/structure-level objectives.
- **Decision.** Obtain pseudo-pairs from an alignment between cells and spots (D5 to D7).
- **Status.** Adopted. The consistency loss of D2 is the only alignment signal before the alignment matrix exists.

### D4. Pseudo-labels from clustering

- **Problem.** Where positives and negatives come from without pairs.
- **Option.** Cluster the latent space and use cluster membership as positives, recomputing clusters periodically (DeepCluster/SwAV style).
- **Reason for rejection.** Clustering scRNA-seq and spatial data jointly is questionable before the modalities are aligned; clustering them separately needs a cluster-matching step; scRNA-seq clusters alone give no spatial positives. Clusters derived from the encoder being trained also make the supervision circular.
- **Status.** Discarded before implementation. A variant using existing annotations is in [section 4](#4-ideas-not-pursued).

### D5. Similarity-derived soft alignment with NT-Xent

- **Problem.** First implemented attempt at cross-modal contrastive learning.
- **What was tried.** Two projection heads $h_{sc}$, $h_{st}$ on top of the latents; cosine similarity matrix between the batches; row-softmax to get a soft alignment; an NT-Xent loss on the result.
- **Why it failed.** Two separate problems. First, with unpaired batches the diagonal is not a positive, and a target computed from the model's own similarity matrix gives no external signal: the model can satisfy it by reproducing its current distribution. Second, NT-Xent weights the most similar negatives the most, and here those are the biologically similar cells that should not be pushed away (see [3.3](#33-why-nt-xent-is-a-poor-fit)).
- **Status.** Discarded. Reported as the first failed alternative in the report's conclusions.

### D6. Learnable batch-level alignment matrix

- **What was tried.** A learnable matrix of size $B_{st} \times B_{sc}$, row-softmaxed, projecting the scRNA-seq batch onto the spatial batch.
- **Why it failed.** The matrix is indexed by batch position. With shuffled loaders, row $i$ is a different cell at every step, so the parameter cannot accumulate a consistent alignment; varying cell-type composition across batches made it worse.
- **Status.** Discarded. Led directly to D7.

### D7. Global alignment matrix indexed by cell and spot ids

- **Decision.** One trainable matrix of logits $\tilde M$ for all cells and all spots. Datasets return each item with its global index, the collate functions keep the indices, and a batch uses the block $M[\text{cell ids}][:, \text{spot ids}]$.
- **Key points.** The softmax is taken over the full row before slicing ([3.2](#32-softmax-over-the-full-row-then-slice)). Index tracking belongs to the dataset, not the encoder. Shuffling is harmless because the block is addressed by global ids.
- **Initialisation.** Logits are drawn from $\mathcal{N}(0, 1)$; a much smaller scale (0.01), giving near-uniform rows at the start, was also considered.
- **Cost.** $n_{sc} \times n_{st}$ parameters: 284 million (about 1.1 GB) for 28,390 cells and 10,000 spots.
- **Status.** Adopted.

### D8. Tangram objective for the alignment

- **Decision.** Train $\tilde M$ with the unconstrained Tangram loss: gene-wise and spot-wise cosine similarity between $M^\top S$ and $G$, an entropy regulariser on the rows, and a KL term between the spot density and a uniform prior.
- **Where.** Either on the latents ($Z_{sc}$, $Z_{st}$) or on the shared genes. Both variants were kept for comparison.
- **Observations.** In gene space both cosine terms converge near 1. The density term increased in both variants, suggesting the uniform prior is a poor fit; the entropy term appeared non-essential. The latent variant plateaued much lower, but that run was affected by a bug (see [section 6](#6-known-issues-and-changes), issue 1), so the comparison has to be repeated.
- **Status.** Adopted. A comparison against the Tangram package itself was started but not completed.

### D9. Contrastive loss: from filtered triplet to thresholded siamese

- **Setup.** Once $M$ exists, row $i$ of the pseudo-spots $M^\top Z_{sc}$ is the positive of spot $i$; the other pseudo-spots of the batch are candidate negatives, some of them biologically similar to the spot.
- **First version.** A triplet loss on the negatives with similarity below a threshold: $\max(0,\ s_{ij} - s_i^+ + m)$. In early runs the positive similarity decreased over the first epochs (0.31 to 0.19) while the contrastive loss stayed flat. The margin also meant that a negative already far from the anchor received no gradient, which was not the intended behaviour.
- **Final version.** For each spot: $(1 - s_i^+)^2$ plus the mean of $\max(0,\ \tau - s_{ij})^2$ over the batch. Negatives with similarity above $\tau$ are ignored.
- **Monitoring.** Mean positive similarity, mean similarity of the selected negatives, and number of selected negatives per epoch. Training stops after a number of epochs without selected negatives.
- **Status.** Adopted; the triplet version is superseded. Open point from the report: penalise negatives in proportion to their similarity instead of treating all selected negatives equally.

### D10. Three-phase training

- **Problem.** Training autoencoders, alignment and contrastive loss together in one phase was unstable. The first implementation was a single model with every loss, the alignment loss included, active from the first epoch.
- **Decision.** (1) Autoencoder initialisation with reconstruction and consistency only. (2) Alignment initialisation: encoders frozen, only $\tilde M$ trained, full batch. (3) Contrastive fine-tuning of the autoencoders, continuing from phases 1 and 2.
- **Reason.** Contrastive learning should refine a latent space that already has structure, not create it while the alignment is also being discovered. Learning $\tilde M$ on frozen representations avoids a feedback loop between projection and contrastive loss.
- **Status.** Adopted. In the package each phase is a LightningModule (`src/spaconae/stages/`).

### D11. Keeping the alignment during fine-tuning: EWC

- **Problem.** In phase 3 the encoders change, so the alignment learned in phase 2 drifts out of date; recomputing the full alignment loss at every mini-batch step is expensive.
- **Options.** Freeze $M$; keep the full-data alignment loss active in phase 3; let the contrastive loss update $M$ under a continual-learning penalty.
- **Decision.** Optional refinement of $\tilde M$ with an EWC penalty $\sum_{ij} \omega_{ij} (\tilde M_{ij} - \tilde M^*_{ij})^2$, with importance $\omega$ from squared gradients of the alignment loss (no likelihood, hence no Fisher information). With the gene-space alignment $M$ stays frozen, because its feature space does not match the latent one.
- **Not constrained.** The autoencoder weights: phase 1 is a short initialisation and reconstruction stays active in phase 3.
- **Status.** Adopted, but the notebook implementation was ineffective (issue 3 in [section 6](#6-known-issues-and-changes)).

### D12. Evaluation by label transfer

- **Decision.** Freeze the encoders, train a small classifier on scRNA-seq latents with the scRNA-seq labels, test it on spatial latents. UMAP of the joint latent space coloured by modality as a qualitative check.
- **Result in the report.** 83.24% accuracy, macro-F1 0.80 on the 10,000 spots.
- **Caveat.** There is no baseline, so the contribution of each phase to that number is unknown ([section 5](#5-evaluation-checklist)).
- **Status.** Adopted; to be extended.

### D13. Refactor into a package

- **Decision.** Port the notebooks to an installable package with PyTorch Lightning, first reproducing their behaviour (commit `76dd3f0`), then fixing the known bugs in separate commits. Dataset-specific preprocessing lives in `scripts/`, and the package only defines an input format (commit `b855c26`).
- **Status.** Adopted.

## 3. Design notes that still hold

### 3.1 Row and column cosine terms are not redundant

Cosine similarity ignores scale. If every row of the prediction is a rescaled copy of the target row, $\hat G_{i,:} = \alpha_i G_{i,:}$, all row cosines equal 1 whatever the $\alpha_i$. The columns, $[\alpha_1 G_{1j}, \dots, \alpha_n G_{nj}]$, are then not parallel to the target columns unless the $\alpha_i$ are constant. Only a global rescaling satisfies both terms. The two terms therefore constrain complementary aspects, and neither is equivalent to a Frobenius loss, which also matches magnitudes.

### 3.2 Softmax over the full row, then slice

$M_{\text{batch}}$ is taken from the row-softmax over all spots. Applying the softmax to the batch block alone would renormalise each cell over the spots that happen to be in the batch, which is a different quantity from the global cell-to-spot distribution.

### 3.3 Why NT-Xent is a poor fit

In $-\log \frac{e^{s^+/t}}{\sum_j e^{s_j/t}}$ the gradient weight of a negative grows with its similarity, and a lower temperature sharpens this. That is the intended hardness-aware behaviour of the loss, and the opposite of what is needed here, where highly similar "negatives" are likely the same cell type. Temperature can flatten the weighting but cannot invert it.

### 3.4 When freezing is needed

A component needs explicit freezing when it is executed but must not update, not merely because it exists. In phase 1 the alignment matrix is never used, so it gets no gradient. In phase 2 the encoders are executed to produce the latents, so they must be frozen (or run without gradients). The phases should be separated by what is computed, not by zeroing losses after a full forward pass.

### 3.5 Loss magnitudes are not gradient contributions

A small contrastive loss next to a large reconstruction loss says little about which one drives the encoders. The comparison that matters is between gradient norms on the shared encoder parameters, $\lVert \nabla_\theta L_{rec} \rVert$ against $\lVert \nabla_\theta L_{con} \rVert$.

### 3.6 Projection heads

If a projection head is used, the contrastive loss acts on its output and the encoder latent stays the representation for downstream use; the head specialises for the contrastive objective. Separate heads per modality are the reasonable default, a shared head being a stronger constraint to test as an ablation.

### 3.7 Five objectives that are easy to conflate

"Integrating scRNA-seq and spatial data" covers different problems, and a method for one does not solve the others: (1) explicit matrix factorisation with non-negativity (NMF, iNMF); (2) latent representation learning, where the latent need not be a factor model; (3) cross-modal alignment of two latents; (4) spatial mapping or deconvolution, i.e. estimating $M$ or cell-type proportions; (5) imputation of unmeasured genes. SpaConAE addresses (2) to (4); imputation would need the scRNA-seq genes outside the shared panel.

### 3.8 Logging pitfalls of the contrastive loss

- The mean positive similarity must be computed over all anchors. Computing it only over anchors that have selected negatives makes it depend on the threshold, although the positives themselves do not.
- When no negative is selected, the mean negative similarity is undefined. Reporting 0 is misleading; NaN is the honest value. The current loss still logs 0 in that case, so a negative similarity of 0 has to be read together with the number of negatives.
- With a margin-based loss, the number of selected negatives is not the number contributing gradient; the fraction of active terms is the informative quantity.

## 4. Ideas not pursued

| Idea | What it would be | Why it was set aside |
|---|---|---|
| NMF / metagene latent | Non-negative latent and decoder, so that latents are factor activations and decoder weights are gene loadings | Different objective (interpretable factors); contrastive loss alone does not give an NMF interpretation |
| iNMF-style shared and private factors | Shared plus modality-specific latent parts | Not needed for a first shared-space model |
| $M$ computed from embeddings | $M = \mathrm{softmax}(\mathrm{sim}(Z_{sc}, Z_{st}))$, attention-like, no free matrix | Needs a non-circular training signal (see D5); would remove the $n_{sc} \times n_{st}$ parameter |
| Cluster-informed contrastive learning | Positives and negatives from existing scRNA-seq annotations; for spots, clusters from expression alone (HVG, PCA, Leiden) or from clustering NMF activations | Spot labels inferred through the alignment itself would be circular; kept as a possible auxiliary loss |
| Distribution-level alignment | Matching the two latent distributions (MMD, Wasserstein, adversarial), or cycle consistency between latent maps $A \to B$ and $B \to A$ | The usual options for unpaired data; MMD is the one stAI uses and was dropped with its other objectives (D2) in favour of pseudo-pairs |
| Projection in the other direction | Each cell as a mixture of spot latents, $M Z_{st}$, with cells as anchors | Discussed early; projecting cells onto spots matches the Tangram formulation |
| Keeping individual cells vs. metacells | Compressing cells before mapping | Discards the fine-grained information the mapping needs; the cell stays the mapping unit |
| Projection heads before the contrastive loss | See 3.6 | Used in D5 only; the final model contrasts latents directly |
| Alignment loss active in phase 3 | Full-data Tangram loss at every fine-tuning step | Cost; replaced by EWC (D11) |
| CNN spatial encoder | Convolutions on a rasterised tissue | MERFISH cells have irregular coordinates; a graph is the natural structure. CNNs suit grid-based data such as Visium |
| Cross-reconstruction as a validation metric | Decode one modality's latent with the other decoder, without gradients, on held-out data | Needs a valid pairing to compare against; as a training loss it is the consistency term |
| Alignment on raw counts | Tangram reportedly works best on raw counts | Not explored (listed as future work in the report) |
| Other datasets | osmFISH with Zeisel, seqFISH, Slide-seqV2, squidpy mouse cortex | Explored only at data-loading level |

## 5. Evaluation checklist

- [x] Label transfer scRNA-seq to spatial (accuracy, macro-F1, per-class report; confusion matrix still to add)
- [x] UMAP of the joint latent space coloured by modality
- [ ] Baselines for the label transfer: raw shared genes, phase-1 encoders only, Tangram
- [ ] Ablations: without contrastive loss, frozen vs. refined $M$, latent vs. gene-space alignment
- [ ] Held-out split for the spatial data
- [ ] UMAP coloured by cell type and metadata, tracked during training
- [ ] Clustering agreement with reference labels (ARI, NMI)
- [ ] Modality-mixing metrics
- [ ] Cross-modal nearest-neighbour retrieval (precision@k) using cell-type labels
- [ ] Alignment and uniformity of the latent space
- [ ] Marker-gene enrichment in latent clusters
- [ ] Spatial coherence of the spatial latents
- [ ] Gradient-norm diagnostics per loss term (3.5)
- [ ] Generalisation to another dataset pair

UMAP is a qualitative check only. The desired picture is modalities mixed while cell types stay separate; complete mixing with biological structure lost is over-alignment, not success.

## 6. Known issues and changes

### Bugs found in the notebooks during the refactor

All five affect the results in the report, which therefore describe a different model from the intended one. They are reproduced in commit `76dd3f0` and fixed one per commit.

| # | Issue | Effect | Fix |
|---|---|---|---|
| 1 | Latent alignment fitted on untrained encoders: the data were encoded when the model was built, before the pretrained weights were loaded, with dropout on | $M$ was optimised against random projections; probably explains the low plateau of the latent alignment (D8) | `217c095` |
| 2 | Contrastive loss returned the value of the last spot of the batch only | One pair per batch contributed to the gradient | `1eb477c` |
| 3 | EWC penalty compared softmaxed $M$ with reference logits, the reference shared memory with the parameter, the importance was one squared gradient at the optimum, $\lambda$ was applied twice | Penalty around $3 \times 10^{-7}$: no constraint on $M$ | `00a99c2` |
| 4 | Decoders re-initialised in phase 3 (only encoders were loaded) | Reconstruction and consistency restarted from random decoders | `ad60c0c` |
| 5 | Early stopper of phase 2 re-created at every epoch | Never triggered; alignment always ran the maximum number of epochs | `3f809ee` |

Two choices made in the fixes are provisional: the EWC importance is now the mean squared gradient accumulated over phase 2, and the early-stopping thresholds (patience 100, 0.1% relative) were guessed from the notebook loss curves.

### Open design weaknesses

Not bugs; candidates for the next round of changes.

- **Composition of the positives.** A pseudo-spot is built from the cells of the current scRNA-seq batch only (512 of 28,390), not from the cells $M$ assigns to that spot. Positives are therefore mostly mixtures of unrelated cells.
- **Spatial graph.** In training the k-nearest-neighbour graph is rebuilt among the random spots of each mini-batch; at inference it is built on the whole patch with a different k.
- **Spatial patch.** It is selected on 2D coordinates although the MERFISH data span several Bregma sections. Whether the patch mixes sections has not been checked.
- **Preprocessing order.** scRNA-seq counts are restricted to the 155 shared genes before total-count normalisation, so library size is computed on the panel only. Discarding the other genes also rules out imputation.
- **Scaling.** The free alignment matrix grows as $n_{sc} \times n_{st}$.
- **Saved configuration.** The contrastive notebook as committed uses $\tau = 0.7$ and 20 epochs; the report states $\tau = 0.88$ and 30 epochs.

### Change log

| Date | Change |
|---|---|
| 2026-11-13 | Package created from the notebooks; five bug fixes; preprocessing moved to `scripts/`; this log written |

## 7. Literature map

Methods mentioned in the discussions, grouped by what they do. The list was collected during the design discussions and has **not been verified** against the original papers.

| Group | Methods | Relation to SpaConAE |
|---|---|---|
| Mapping by constrained projection | Tangram | Source of the alignment objective |
| Shared-latent autoencoders | stAI, stPlus; deep CCA variants (DCCA, DCCAE) | stAI is the architectural starting point |
| Probabilistic generative models | gimVI (scvi-tools), Stereoscope | Alternative approach to integration, imputation and deconvolution |
| Matrix factorisation | NMF, iNMF / LIGER, INSPIRE, FAST | Considered and set aside (D1) |
| Spatial representation with graphs | SEDR, STdGCN, GraphST, scVGAE (scRNA-seq) | Related use of spatial neighbourhoods |
| Contrastive learning on spatial data | GraphST, CellContrast | Contrastive objectives, different pairing strategy |
| Deep clustering of scRNA-seq | scDeepCluster, DESC | Background for the clustering idea (D4) |
| Transformers and diffusion for imputation | SpaFormer, stEnTrans, SpaDiT, stMCDI | Not used |
| Other mapping and deconvolution models | MACD, STALocator, DTractor | Not used |
| Latent-space quality metrics | Alignment and uniformity (Wang and Isola) | Candidate evaluation metric |
