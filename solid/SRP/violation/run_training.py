"""Entry point for the SRP violation example."""

import argparse
from .src.model_trainer import ModelTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="Path to model config YAML")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, dest="learning_rate")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_stratify", action="store_true")
    args = parser.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    if args.no_stratify:
        overrides["use_stratify"] = False
    ModelTrainer(config_path=args.config, overrides=overrides).run()


if __name__ == "__main__":
    main()


