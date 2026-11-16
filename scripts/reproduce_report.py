"""Re-run the analysis of the project report and compare two code versions.

Runs, on the Moffitt data, the pipeline of the report twice from the same
pretrained autoencoders: once with the code as ported from the notebooks
(``--legacy-rev``, known bugs included) and once with the bugs fixed
(``--fixed-rev``). Both are fixed commits, checked out in the work dir, so the
result does not depend on later changes to the model. It also evaluates the
checkpoints of the original run when they are available.

Outputs (small files only) go to ``--out-dir``: one log per stage, the metric
CSVs, ``progress.json`` and ``results.json``. Large files (alignment matrices)
stay in ``--work-dir``. Finished stages are skipped when the script is re-run.

The work can be split over sessions. ``--stages gpu`` runs only the training
phases; ``--stages cpu`` runs only the probes and the analyses, which need no
GPU. With ``--keep-alignment`` a finished alignment (about 2 GB) is copied to
the output dir until its fine-tuning is done, so that an interrupted session
does not have to repeat it.

Example (Colab, Drive mounted)::

    python scripts/reproduce_report.py \\
        --data-dir "$DRIVE/data" --orig-ckpt-dir "$DRIVE/models_checkpoints" \\
        --out-dir "$DRIVE/reproduction" --work-dir /content/work
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
ST_FILE, SC_FILE = "moffit_spatial_train.h5ad", "moffit_scRNA_train.h5ad"
WEIGHTS = ["spatial_encoder", "scrna_encoder", "spatial_decoder", "scrna_decoder"]


# --------------------------------------------------------------------------- utilities

class Run:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out_dir)
        self.work = Path(args.work_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        self.results_file = self.out / "results.json"
        self.results = json.loads(self.results_file.read_text()) if self.results_file.exists() else {}
        self.legacy_tree = self.work / "legacy_src"
        self.fixed_tree = self.work / "fixed_src"

    def save(self, key, value):
        self.results[key] = value
        self.results_file.write_text(json.dumps(self.results, indent=1, default=float))

    def progress(self, stage, status, **extra):
        entry = {"stage": stage, "status": status, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
        log = self.results.get("progress", [])
        log.append(entry)
        self.save("progress", log)
        (self.out / "progress.json").write_text(json.dumps(log, indent=1))
        print(f"[{entry['time']}] {stage}: {status} {extra if extra else ''}", flush=True)

    def done(self, stage):
        return any(p["stage"] == stage and p["status"] == "done" for p in self.results.get("progress", []))

    def forget(self, stage):
        """Drop the record of a finished stage whose output is gone, so it runs again."""
        self.save("progress", [p for p in self.results.get("progress", []) if p["stage"] != stage])

    def keep(self, ckpt, prefix, suffix):
        """Copy the four weight files to the output dir (or back, after the work dir was lost)."""
        Path(ckpt).mkdir(parents=True, exist_ok=True)
        for w in WEIGHTS:
            local, kept = Path(ckpt) / f"{w}_weights{suffix}.pth", self.out / f"{prefix}_{w}_weights{suffix}.pth"
            if local.exists():
                shutil.copy(local, kept)
            elif kept.exists():
                shutil.copy(kept, local)

    def kept_alignment(self, name):
        return self.out / f"{name}_ewc_alignment.pt"

    def cli(self, stage_name, tree, stage, cfg):
        """Run one pipeline stage of the code in ``tree`` in a subprocess."""
        cfg_file = self.work / f"{stage_name}.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
        env = dict(os.environ, PYTHONPATH=str(Path(tree) / "src"), PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, "-m", "spaconae.cli", stage, "--config", str(cfg_file)]
        t0 = time.time()
        with open(self.out / f"{stage_name}.log", "w") as log:
            code = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return code, round((time.time() - t0) / 60, 1)

    def stage(self, stage_name, tree, stage, cfg, fallback=None):
        if self.done(stage_name):
            return True
        self.progress(stage_name, "started")
        code, minutes = self.cli(stage_name, tree, stage, cfg)
        note = None
        if code != 0 and fallback is not None:
            note = fallback(cfg)
            self.progress(stage_name, "retry", note=note)
            code, minutes = self.cli(stage_name, tree, stage, cfg)
        if code != 0:
            self.progress(stage_name, "FAILED", minutes=minutes)
            return False
        self.copy_metrics(stage_name, cfg["log_dir"], stage)
        self.progress(stage_name, "done", minutes=minutes, **({"note": note} if note else {}))
        return True

    def copy_metrics(self, stage_name, log_dir, stage):
        files = sorted(Path(log_dir).glob(f"{stage}/version_*/metrics.csv"))
        if files:
            shutil.copy(files[-1], self.out / f"{stage_name}_metrics.csv")
        report = Path(log_dir) / "probe" / "classification_report.txt"
        if stage == "probe" and report.exists():
            shutil.copy(report, self.out / f"{stage_name}_report.txt")

    def config(self, tree, name, legacy):
        """The repository config of ``tree`` with paths and report settings applied."""
        cfg = yaml.safe_load((Path(tree) / "configs" / "moffitt.yaml").read_text())
        cfg.pop("preprocess", None)
        if legacy:
            cfg["data_dir"] = str(self.a.data_dir)
        else:
            cfg["data"]["dir"] = str(self.a.data_dir)
        cfg["checkpoint_dir"] = str(self.work / name / "ckpt")
        cfg["log_dir"] = str(self.work / name / "logs")
        cfg["progress_bar"] = False
        # Final model of the report: tau = 0.88, 30 epochs
        cfg["finetune"].update(sim_threshold=self.a.tau, max_epochs=self.a.finetune_epochs, save_alignment=False)
        if self.a.smoke:
            cfg["model"].update(d_hidden=16, d_latent=8)
            cfg["pretrain"].update(max_epochs=2, batch_size_st=16, batch_size_sc=16, knn=4)
            cfg["align"].update(max_epochs=5, knn=5)
            cfg["finetune"].update(max_epochs=2, batch_size_st=16, batch_size_sc=32, knn=4)
            cfg["probe"].update(max_epochs=2, batch_size_st=16, knn=5)
        return cfg


def metric_summary(csv_file):
    """First, last, minimum (with epoch) of every logged metric."""
    import csv

    cols = {}
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            for k, v in row.items():
                if k not in ("epoch", "step") and v not in ("", None):
                    cols.setdefault(k, []).append((int(row["epoch"]), float(v)))
    out = {}
    for k, vals in cols.items():
        v = [x for _, x in vals]
        i = int(np.nanargmin(v)) if not all(np.isnan(v)) else 0
        out[k] = {"first": v[0], "last": v[-1], "min": v[i], "min_epoch": vals[i][0], "n_epochs": len(v)}
    return out


# --------------------------------------------------------------------------- analyses

def load_data(run):
    sys.path.insert(0, str(run.fixed_tree / "src"))
    from spaconae.data import load_pair

    return load_pair(run.a.data_dir, ST_FILE, SC_FILE, label_key="Cell_class")


def encode(run, data, ckpt_dir, suffix, model_cfg):
    """Latents of both full datasets with the encoders in evaluation mode."""
    from spaconae import CrossModalAE

    ae = CrossModalAE(d_input=data.n_genes, **model_cfg).load_weights(ckpt_dir, suffix=suffix, decoders=False).eval()
    full_sc, full_st, edge = data.full_batch(knn=10)
    z_sc, z_st = ae.encode(full_sc, full_st, edge)
    return z_sc.numpy(), z_st.numpy()


def latent_metrics(z_sc, z_st, y_sc, y_st, k=15, max_cells=10000, seed=0):
    """Classifier-free measures of how well the two latents overlap."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    # kNN label transfer: each spot takes the majority label of its k nearest cells
    nn_sc = NearestNeighbors(n_neighbors=k).fit(z_sc)
    idx = nn_sc.kneighbors(z_st, return_distance=False)
    n_classes = int(max(y_sc.max(), y_st.max())) + 1
    votes = np.apply_along_axis(lambda r: np.bincount(r, minlength=n_classes).argmax(), 1, y_sc[idx])
    knn_acc = float((votes == y_st).mean())

    # Modality mixing: share of cross-modality neighbours, on equal-sized samples
    n = min(len(z_sc), len(z_st), max_cells)
    a = z_sc[rng.choice(len(z_sc), n, replace=False)]
    b = z_st[rng.choice(len(z_st), n, replace=False)]
    joint = np.vstack([a, b])
    modality = np.r_[np.zeros(n), np.ones(n)]
    nb = NearestNeighbors(n_neighbors=k + 1).fit(joint).kneighbors(joint, return_distance=False)[:, 1:]
    mixing = float((modality[nb] != modality[:, None]).mean())  # 0.5 = perfectly mixed, 0 = separate

    clf = LogisticRegression(max_iter=2000).fit(z_sc, y_sc)
    return {"knn_label_transfer_acc": knn_acc, "modality_mixing": mixing,
            "linear_probe_acc": float(clf.score(z_st, y_st))}


def alignment_metrics(M_logits, y_sc, y_st, chunk=2000):
    """Does the alignment matrix map cells to spots of the same cell type?

    Runs on the CPU in chunks of rows: this process must not hold GPU memory,
    which the training subprocesses need in full.
    """
    y_sc_t, y_st_t = torch.as_tensor(y_sc), torch.as_tensor(y_st)
    n_classes = int(max(y_sc.max(), y_st.max())) + 1
    onehot = torch.nn.functional.one_hot(y_sc_t, n_classes).float()
    n_cells, n_spots = M_logits.shape
    top_spot = torch.empty(n_cells, dtype=torch.long)
    max_p = torch.empty(n_cells)
    entropy = torch.empty(n_cells)
    spot_mass = torch.zeros(n_spots, n_classes)
    with torch.no_grad():
        for i in range(0, n_cells, chunk):
            M = torch.softmax(M_logits[i:i + chunk].float(), dim=1)
            max_p[i:i + chunk], top_spot[i:i + chunk] = M.max(1)
            entropy[i:i + chunk] = -(M * torch.log(M + 1e-30)).sum(1)
            spot_mass += M.t() @ onehot[i:i + chunk]
    # chance level for the cell -> spot agreement
    p_st = np.bincount(y_st, minlength=n_classes) / len(y_st)
    p_sc = np.bincount(y_sc, minlength=n_classes) / len(y_sc)
    return {
        # cell -> its most probable spot
        "cell_top_spot_same_type": (y_st_t[top_spot] == y_sc_t).float().mean().item(),
        # spot -> label with the largest probability mass (Tangram-style annotation projection)
        "spot_label_projection_acc": (spot_mass.argmax(1) == y_st_t).float().mean().item(),
        "mean_max_prob": max_p.mean().item(),
        "mean_row_entropy": entropy.mean().item(),
        "uniform_entropy": float(np.log(n_spots)),
        "chance_cell_top_spot": float((p_sc * p_st).sum()),
        "majority_class_spot_acc": float(p_st.max()),
    }


def load_M(path):
    from spaconae.nn import LEGACY_KEY

    return torch.load(path, map_location="cpu")["optimal_params"][LEGACY_KEY]


def save_umap(run, name, z_sc, z_st, y_sc, y_st, classes):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap
    except ImportError:
        return False
    rng = np.random.default_rng(0)
    keep = rng.choice(len(z_sc), min(len(z_sc), len(z_st)), replace=False)
    z = np.vstack([z_sc[keep], z_st])
    emb = umap.UMAP(n_neighbors=15, random_state=0).fit_transform(z)
    modality = np.r_[np.zeros(len(keep)), np.ones(len(z_st))]
    labels = np.r_[y_sc[keep], y_st]
    order = rng.permutation(len(z))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].scatter(*emb[order].T, c=modality[order], cmap="coolwarm", s=1, alpha=0.5)
    axes[0].set_title(f"{name}: modality (blue scRNA-seq, red spatial)")
    sc_ = axes[1].scatter(*emb[order].T, c=labels[order], cmap="tab10", s=1, alpha=0.5)
    axes[1].legend(sc_.legend_elements()[0], list(classes), fontsize=7, markerscale=2)
    axes[1].set_title(f"{name}: cell type")
    fig.tight_layout()
    fig.savefig(run.out / f"umap_{name}.png", dpi=110)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--orig-ckpt-dir", help="checkpoints of the original run (optional)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--legacy-rev", default="76dd3f0", help="commit reproducing the notebooks")
    p.add_argument("--fixed-rev", default="c7ba257", help="commit with the bugs of the notebooks fixed")
    p.add_argument("--tau", type=float, default=0.88)
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--stages", choices=["all", "gpu", "cpu"], default="all",
                   help="gpu: training phases only; cpu: probes and analyses only")
    p.add_argument("--keep-alignment", action="store_true",
                   help="copy each alignment to the output dir until its fine-tuning is done")
    p.add_argument("--smoke", action="store_true", help="tiny settings, to test the script")
    run = Run(p.parse_args())
    a = run.a
    train, analyse = a.stages in ("all", "gpu"), a.stages in ("all", "cpu")
    # Query the GPU without creating a CUDA context in this process
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None

    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    for tree, rev in ((run.legacy_tree, a.legacy_rev), (run.fixed_tree, a.fixed_rev)):
        if not tree.exists():
            subprocess.check_call(["git", "-C", str(REPO), "worktree", "add", "--detach", str(tree), rev])
    run.save("environment", {
        "head": head, "legacy_rev": a.legacy_rev, "fixed_rev": a.fixed_rev, "torch": torch.__version__, "gpu": gpu,
        "tau": a.tau, "finetune_epochs": a.finetune_epochs, "smoke": a.smoke,
    })

    data = load_data(run)
    run.save("data", {
        "n_cells": data.n_cells, "n_spots": data.n_spots, "n_genes": data.n_genes,
        "classes": list(map(str, data.classes)),
        "spot_class_counts": np.bincount(data.y_st).tolist(),
        "cell_class_counts": np.bincount(data.y_sc).tolist(),
    })

    cfg_new = run.config(run.fixed_tree, "fixed", legacy=False)
    cfg_old = run.config(run.legacy_tree, "legacy", legacy=True)
    model_cfg = cfg_new["model"]
    ckpt_new, ckpt_old = Path(cfg_new["checkpoint_dir"]), Path(cfg_old["checkpoint_dir"])
    embeddings = {}  # name -> (checkpoint dir, suffix)

    def probe(name, ckpt_dir, suffix):
        if not analyse:
            return False
        cfg = json.loads(json.dumps(cfg_new))
        cfg["checkpoint_dir"] = str(ckpt_dir)
        cfg["log_dir"] = str(run.work / name / "logs")
        cfg["probe"]["weights_suffix"] = suffix
        ok = run.stage(f"probe_{name}", run.fixed_tree, "probe", cfg)
        if ok:
            embeddings[name] = (ckpt_dir, suffix)
        return ok

    # 1. Checkpoints of the original run
    if a.orig_ckpt_dir and (Path(a.orig_ckpt_dir) / "scrna_encoder_weights_finetuned.pth").exists() and not a.smoke:
        probe("original_finetuned", a.orig_ckpt_dir, "_finetuned")
        probe("original_pretrained", a.orig_ckpt_dir, "")

    # 2. Phase 1, once; both versions continue from the same autoencoders
    run.keep(ckpt_new, "new", "")  # restores the weights if an earlier session produced them
    if not (ckpt_new / "scrna_encoder_weights.pth").exists():
        if not train:
            raise SystemExit("No pretrained weights yet: run the training phases first (--stages gpu)")
        run.forget("pretrain")
    if not run.stage("pretrain", run.fixed_tree, "pretrain", cfg_new):
        return
    run.keep(ckpt_new, "new", "")
    ckpt_old.mkdir(parents=True, exist_ok=True)
    for w in WEIGHTS:
        shutil.copy(ckpt_new / f"{w}_weights.pth", ckpt_old / f"{w}_weights.pth")
    probe("new_pretrained", ckpt_new, "")

    # 3. Phases 2 and 3 with the notebook behaviour, then with the current code
    def no_accumulation(cfg):
        cfg["align"]["accumulate_importance"] = False
        return "alignment failed (probably out of memory); retried with accumulate_importance=false"

    for name, tree, cfg, ckpt, fallback in (
        ("legacy", run.legacy_tree, cfg_old, ckpt_old, None),
        ("fixed", run.fixed_tree, cfg_new, ckpt_new, no_accumulation),
    ):
        ewc, kept = ckpt / "ewc_alignment.pt", run.kept_alignment(name)
        run.keep(ckpt, name, "_finetuned")
        if not (ckpt / "scrna_encoder_weights_finetuned.pth").exists():
            if not train:
                continue
            run.forget(f"finetune_{name}")
            if not ewc.exists() and kept.exists():
                shutil.copy(kept, ewc)  # alignment finished in an earlier session
            if not ewc.exists():
                run.forget(f"align_{name}")
            if not run.stage(f"align_{name}", tree, "align", cfg, fallback):
                continue
            if a.keep_alignment and not kept.exists():
                shutil.copy(ewc, kept)
            run.save(f"alignment_{name}", alignment_metrics(load_M(ewc), data.y_sc, data.y_st))
            if not run.stage(f"finetune_{name}", tree, "finetune", cfg):
                continue
            run.keep(ckpt, name, "_finetuned")
            ewc.unlink()  # 2 GB, no longer needed
            kept.unlink(missing_ok=True)
        probe(f"{name}_finetuned", ckpt, "_finetuned")

    # 4. Gene-space alignment (the comparison of the report's alignment figure)
    cfg_gene = json.loads(json.dumps(cfg_new))
    cfg_gene["align"].update(encode_data=False, lambda_r=0.0, max_epochs=5 if a.smoke else 700,
                             accumulate_importance=False, output_file="ewc_alignment_noAE.pt")
    cfg_gene["checkpoint_dir"] = str(run.work / "gene" / "ckpt")
    cfg_gene["log_dir"] = str(run.work / "gene" / "logs")
    if train and "alignment_gene_space" not in run.results and run.stage("align_gene_space", run.fixed_tree, "align", cfg_gene):
        f = Path(cfg_gene["checkpoint_dir"]) / "ewc_alignment_noAE.pt"
        run.save("alignment_gene_space", alignment_metrics(load_M(f), data.y_sc, data.y_st))
        f.unlink()

    # 5. Alignment matrices of the original run
    if analyse and a.orig_ckpt_dir and not a.smoke:
        for key, fname in (("alignment_original_latent", "ewc_alignment.pt"), ("alignment_original_gene_space", "ewc_alignment_noAE.pt")):
            f = Path(a.orig_ckpt_dir) / fname
            if key not in run.results and f.exists():
                run.progress(key, "started")
                run.save(key, alignment_metrics(load_M(f), data.y_sc, data.y_st))
                run.progress(key, "done")

    # 6. Latent-space measures and UMAPs, raw-gene baseline
    if analyse:
        from sklearn.linear_model import LogisticRegression

        latent = run.results.get("latent", {})
        if "raw_genes" not in latent:
            clf = LogisticRegression(max_iter=2000).fit(data.X_sc, data.y_sc)
            latent["raw_genes"] = latent_metrics(data.X_sc, data.X_st, data.y_sc, data.y_st)
            latent["raw_genes"]["linear_probe_acc"] = float(clf.score(data.X_st, data.y_st))
            run.save("latent", latent)
        for name, (ckpt_dir, suffix) in embeddings.items():
            if name in latent:
                continue
            z_sc, z_st = encode(run, data, ckpt_dir, suffix, model_cfg)
            latent[name] = latent_metrics(z_sc, z_st, data.y_sc, data.y_st)
            latent[name]["umap_saved"] = save_umap(run, name, z_sc, z_st, data.y_sc, data.y_st, data.classes)
            run.save("latent", latent)

    # 7. Summaries of the logged curves
    run.save("curves", {f.name.replace("_metrics.csv", ""): metric_summary(f) for f in sorted(run.out.glob("*_metrics.csv"))})
    run.progress(f"session ({a.stages})", "done")


if __name__ == "__main__":
    main()
