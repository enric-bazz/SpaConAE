"""Command line entry point: ``spaconae <stage> --config configs/moffitt.yaml``."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="spaconae", description=__doc__)
    parser.add_argument(
        "stage", choices=["pretrain", "align", "finetune", "probe", "all"],
        help="'all' runs pretrain, align, finetune and probe in sequence",
    )
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--data-dir", dest="data_dir", help="folder with the preprocessed .h5ad files")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", help="folder for model weights")
    parser.add_argument("--log-dir", dest="log_dir", help="folder for training logs")
    parser.add_argument("--accelerator", help="Lightning accelerator: auto, cpu, gpu, ...")
    args = parser.parse_args(argv)

    # Imported here so that `spaconae --help` stays fast
    from .pipeline import STAGES, load_config, run_all

    cfg = load_config(
        args.config,
        data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir, accelerator=args.accelerator,
    )

    if args.stage == "all":
        run_all(cfg)
    else:
        STAGES[args.stage](cfg)


if __name__ == "__main__":
    main()
