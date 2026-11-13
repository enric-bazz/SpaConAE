"""The training pipeline: one function per phase, driven by a config dict.

Phases exchange the same files the notebooks wrote to Google Drive
(``*_weights.pth``, ``ewc_alignment.pt``, ...), so each phase can be run on its
own and checkpoints produced by the notebooks remain usable.
"""

from pathlib import Path

import lightning as L
import torch
import yaml
from lightning.pytorch.loggers import CSVLogger

from .callbacks import LossPlateauStopping, NegativeCountStopping
from .data import LabelTransferDataModule, PairedDataModule, load_pair
from .model import CrossModalAE
from .nn import LEGACY_KEY
from .stages import AEPretrainModule, AlignmentModule, ContrastiveModule, LabelTransferModule


def load_config(path, **overrides):
    """Read a YAML config; non-``None`` keyword arguments override its entries."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    data_dir = overrides.pop("data_dir", None)
    if data_dir is not None:
        cfg["data"]["dir"] = data_dir
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _trainer(cfg, stage, max_epochs, callbacks=(), **kwargs):
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator=cfg.get("accelerator", "auto"),
        devices=1,
        logger=CSVLogger(cfg["log_dir"], name=stage),
        # Lightning checkpoints would include M and its optimizer state
        # (several GB); the phases export only the weights they need.
        enable_checkpointing=False,
        callbacks=list(callbacks),
        log_every_n_steps=1,
        enable_progress_bar=cfg.get("progress_bar", True),
        **kwargs,
    )


def _load_data(cfg, labels=False):
    d = cfg["data"]
    return load_pair(d["dir"], d["st_file"], d["sc_file"], label_key=d.get("label_key") if labels else None)


def _new_ae(cfg, data):
    return CrossModalAE(d_input=data.n_genes, **cfg["model"])


def run_pretrain(cfg):
    """Phase 1: train the two autoencoders and save their weights."""
    c = cfg["pretrain"]
    L.seed_everything(cfg["seed"])
    data = _load_data(cfg)
    dm = PairedDataModule(data, batch_size_st=c["batch_size_st"], batch_size_sc=c["batch_size_sc"], knn=c["knn"])

    ae = _new_ae(cfg, data)
    module = AEPretrainModule(
        ae, lambda_recon=c["lambda_recon"], lambda_consistence=c["lambda_consistence"], lr=c["lr"]
    )
    _trainer(cfg, "pretrain", c["max_epochs"]).fit(module, dm)

    ae.save_weights(cfg["checkpoint_dir"])
    return ae


def run_align(cfg):
    """Phase 2: learn the alignment matrix, save it with its EWC importance."""
    c = cfg["align"]
    L.seed_everything(cfg["seed"])
    data = _load_data(cfg)
    full_sc, full_st, full_edge_index = data.full_batch(knn=c["knn"])

    ae = _new_ae(cfg, data)
    if c["encode_data"]:
        # Must happen before the module is built: it encodes the data in __init__
        ae.load_weights(cfg["checkpoint_dir"], decoders=False)
    module = AlignmentModule(
        ae, full_sc, full_st, full_edge_index,
        encode_data=c["encode_data"],
        lambda_g1=c["lambda_g1"], lambda_g2=c["lambda_g2"], lambda_r=c["lambda_r"], lambda_d=c["lambda_d"],
        lr=c["lr"],
        accumulate_importance=c.get("accumulate_importance", True),
    )

    stopper = LossPlateauStopping(
        monitor="total", patience=c["patience"], min_delta=c["min_delta"], relative=c["relative_delta"]
    )
    trainer = _trainer(cfg, "align", c["max_epochs"], callbacks=[stopper])
    trainer.fit(module)

    # The importance needs a full forward/backward pass: use the GPU if there is one
    if trainer.strategy.root_device.type != "cpu":
        module.to(trainer.strategy.root_device)
    out = Path(cfg["checkpoint_dir"]) / c["output_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.ewc_state(), out)
    return module


def run_finetune(cfg):
    """Phase 3: contrastive fine-tuning, starting from phases 1 and 2."""
    c = cfg["finetune"]
    L.seed_everything(cfg["seed"])
    data = _load_data(cfg)
    dm = PairedDataModule(data, batch_size_st=c["batch_size_st"], batch_size_sc=c["batch_size_sc"], knn=c["knn"])

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ewc_info = torch.load(ckpt_dir / c["alignment_file"], map_location="cpu")
    initial_matrix = ewc_info["optimal_params"][LEGACY_KEY]
    importance = ewc_info["importance"][LEGACY_KEY] if c["continual_alignment"] else None

    ae = _new_ae(cfg, data)
    # Continue from the pretrained autoencoders, decoders included
    ae.load_weights(ckpt_dir, decoders=True)

    module = ContrastiveModule(
        ae,
        initial_matrix=initial_matrix,
        importance_matrix=importance,
        continual_alignment=c["continual_alignment"],
        lambda_recon=c["lambda_recon"],
        lambda_contrast=c["lambda_contrast"],
        lambda_ewc=c["lambda_ewc"],
        lambda_consistence=c["lambda_consistence"],
        sim_threshold=c["sim_threshold"],
        lr=c["lr"],
    )
    stopper = NegativeCountStopping(patience=c["negatives_patience"])
    _trainer(cfg, "finetune", c["max_epochs"], callbacks=[stopper]).fit(module, dm)

    ae.save_weights(ckpt_dir, suffix=c["output_suffix"])
    if c["save_alignment"]:
        torch.save(module.AlignMat.state_dict(), ckpt_dir / f"alignment_matrix{c['output_suffix']}.pth")
    return module


def run_probe(cfg):
    """Train the classifier on scRNA-seq latents and test it on spatial latents."""
    c = cfg["probe"]
    L.seed_everything(cfg["seed"])
    data = _load_data(cfg, labels=True)
    dm = LabelTransferDataModule(
        data, batch_size_sc=c["batch_size_sc"], batch_size_st=c["batch_size_st"],
        knn=c["knn"], val_fraction=c["val_fraction"],
    )

    ae = _new_ae(cfg, data)
    ae.load_weights(cfg["checkpoint_dir"], suffix=c["weights_suffix"], decoders=False)

    module = LabelTransferModule(
        ae, n_classes=len(data.classes), classifier_hidden=c["classifier_hidden"],
        lr=c["lr"], class_names=data.classes,
    )
    trainer = _trainer(cfg, "probe", c["max_epochs"], num_sanity_val_steps=0)
    trainer.fit(module, dm)
    results = trainer.test(module, dm)

    print(module.test_report)
    report_file = Path(cfg["log_dir"]) / "probe" / "classification_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(module.test_report)
    return results[0]


STAGES = {
    "pretrain": run_pretrain,
    "align": run_align,
    "finetune": run_finetune,
    "probe": run_probe,
}


def run_all(cfg):
    for stage in STAGES.values():
        stage(cfg)
