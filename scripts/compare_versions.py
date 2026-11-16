"""Compare the current code with an earlier version of the package.

Runs the whole pipeline on the Moffitt data twice: with the baseline
(``--baseline-rev``, by default the port of the notebooks with their bugs
fixed) and with the working tree. Each version uses its own code and its own
``configs/moffitt.yaml``, also to compute the embeddings that are scored, so
that each keeps its own spatial graph. The environment must hold the
dependencies of both versions (the default baseline also needs ``pynndescent``).

``reproduce_report.py`` is a different check (notebook behaviour against the
bug fixes, at the baseline commit) and is left as it was; this script reuses
its helpers and its measures, so the numbers of the two are comparable.

Outputs (small files only) go to ``--out-dir``: one log per stage, the metric
CSVs, ``progress.json`` and ``results.json``. Large files stay in
``--work-dir``. Finished stages are skipped when the script is re-run.

Example (Colab, Drive mounted)::

    python scripts/compare_versions.py \\
        --data-dir "$DRIVE/data" --out-dir "$DRIVE/comparison" --work-dir /content/work
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from reproduce_report import REPO, Run, alignment_metrics, latent_metrics, metric_summary, save_umap

STAGES = ("pretrain", "align", "finetune", "probe")

# Run with the code of one version: both full datasets through its encoders
EXPORT = """
import sys, yaml, numpy as np
from spaconae import CrossModalAE
from spaconae.pipeline import _load_data

cfg_file, suffix, out = sys.argv[1:]
cfg = yaml.safe_load(open(cfg_file))
data = _load_data(cfg, labels=True)
ae = CrossModalAE(d_input=data.n_genes, **cfg["model"])
ae.load_weights(cfg["checkpoint_dir"], suffix=suffix, decoders=False).eval()
z_sc, z_st = ae.encode(*data.full_batch())
np.savez(out, z_sc=z_sc.numpy(), z_st=z_st.numpy(), y_sc=data.y_sc, y_st=data.y_st, classes=data.classes.astype(str))
"""


def export_latents(run, name, tree, cfg, suffix):
    """Latents and labels of both datasets, computed by the code in ``tree``."""
    cfg_file = run.work / f"{name}_export.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    out = run.work / name / f"latents{suffix}.npz"
    env = dict(os.environ, PYTHONPATH=str(Path(tree) / "src"), CUDA_VISIBLE_DEVICES="")
    subprocess.check_call([sys.executable, "-c", EXPORT, str(cfg_file), suffix, str(out)], env=env)
    return np.load(out)


def no_accumulation(cfg):
    cfg["align"]["accumulate_importance"] = False
    return "alignment failed (probably out of memory); retried with accumulate_importance=false"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--baseline-rev", default="2b685e9", help="commit of the version to compare against")
    p.add_argument("--tau", type=float, default=0.88)
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--smoke", action="store_true", help="tiny settings, to test the script")
    run = Run(p.parse_args())
    a = run.a

    baseline_tree = run.work / "baseline_src"
    if not baseline_tree.exists():
        subprocess.check_call(["git", "-C", str(REPO), "worktree", "add", "--detach", str(baseline_tree), a.baseline_rev])
    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    run.save("environment", {
        "head": head, "baseline_rev": a.baseline_rev, "torch": torch.__version__,
        "tau": a.tau, "finetune_epochs": a.finetune_epochs, "smoke": a.smoke,
    })

    latent = run.results.get("latent", {})
    for name, tree in (("baseline", baseline_tree), ("current", REPO)):
        cfg = run.config(tree, name, legacy=False)
        ckpt = Path(cfg["checkpoint_dir"])
        matrix = ckpt / cfg["align"]["output_file"]

        for stage in STAGES:
            if stage == "align" and not matrix.exists() and not run.done(f"finetune_{name}"):
                run.forget(f"align_{name}")  # the matrix lives in the work dir only
            if not run.stage(f"{stage}_{name}", tree, stage, cfg, no_accumulation if stage == "align" else None):
                return
        report = run.out / f"probe_{name}_report.txt"
        run.save(f"probe_{name}", report.read_text() if report.exists() else None)

        for label, suffix in (("pretrained", ""), ("finetuned", cfg["probe"]["weights_suffix"])):
            key = f"{name}_{label}"
            score_matrix = label == "pretrained" and f"alignment_{name}" not in run.results
            if key in latent and not score_matrix:
                continue
            z = export_latents(run, name, tree, cfg, suffix)
            if key not in latent:
                latent[key] = latent_metrics(z["z_sc"], z["z_st"], z["y_sc"], z["y_st"])
                latent[key]["umap_saved"] = save_umap(run, key, z["z_sc"], z["z_st"], z["y_sc"], z["y_st"], z["classes"])
                run.save("latent", latent)
            if score_matrix:
                M = next(iter(torch.load(matrix, map_location="cpu")["optimal_params"].values()))
                run.save(f"alignment_{name}", alignment_metrics(M, z["y_sc"], z["y_st"]))
                matrix.unlink()  # 2 GB, no longer needed

    run.save("curves", {f.name.replace("_metrics.csv", ""): metric_summary(f) for f in sorted(run.out.glob("*_metrics.csv"))})
    run.progress("all", "done")

    print(f"\n{'':22}{'kNN transfer':>14}{'linear probe':>14}{'mixing':>10}")
    for key, m in latent.items():
        print(f"{key:22}{m['knn_label_transfer_acc']:14.3f}{m['linear_probe_acc']:14.3f}{m['modality_mixing']:10.3f}")


if __name__ == "__main__":
    main()
