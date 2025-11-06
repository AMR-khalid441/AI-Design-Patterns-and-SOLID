"""Entry point for the SRP solution example."""

import argparse
from .src.main import load_config, build_pipeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=False, help="Path to model config YAML")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float, dest="learning_rate")
    p.add_argument("--batch_size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--no_stratify", action="store_true")
    args = p.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    if args.no_stratify:
        overrides["use_stratify"] = False

    cfg = load_config(args.config, overrides)
    pipeline = build_pipeline(cfg)
    meta = pipeline.run()
    print(meta)


if __name__ == "__main__":
    main()

"""Entry point for the SRP solution example (stub)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="Path to model config YAML")
    _ = parser.parse_args()
    print("[solution] training stub")


if __name__ == "__main__":
    main()


