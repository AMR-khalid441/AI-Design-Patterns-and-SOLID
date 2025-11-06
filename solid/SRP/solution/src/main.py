"""Composition root for SRP solution."""

import logging
import random
import yaml
import numpy as np
import torch
from pathlib import Path
from .domain.training_config import TrainingConfig
from .services.data.csv_loader import CsvLoader
from .services.training.models.simple_mlp import SimpleMLP
from .services.training.pytorch_trainer import PyTorchTrainer
from .services.training.model_evaluator import ModelEvaluator
from .services.storage.local_model_repository import LocalModelRepository
from .services.tracking.console_tracker import ConsoleTracker
from .pipelines.training_pipeline import TrainingPipeline


def load_config(config_path: str | None, overrides: dict | None = None) -> TrainingConfig:
    cfg = {
        "learning_rate": 0.001,
        "epochs": 5,
        "hidden_sizes": [16, 8],
        "batch_size": 16,
        "seed": 42,
        "test_size": 0.2,
        "random_state": 42,
        "use_stratify": True,
    }
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                cfg.update(loaded)
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return TrainingConfig(
        learning_rate=float(cfg["learning_rate"]),
        epochs=int(cfg["epochs"]),
        hidden_sizes=list(cfg["hidden_sizes"]),
        batch_size=int(cfg["batch_size"]),
        seed=int(cfg["seed"]),
        test_size=float(cfg["test_size"]),
        random_state=int(cfg["random_state"]),
        use_stratify=bool(cfg["use_stratify"]),
    )


def build_pipeline(config: TrainingConfig):
    logging.basicConfig(level=logging.INFO, format="[solution] %(message)s")
    # seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    loader = CsvLoader(
        seed=config.seed,
        test_size=config.test_size,
        random_state=config.random_state,
        use_stratify=config.use_stratify,
    )
    # build model using input_dim=4 for Iris and num_classes=3
    model = SimpleMLP(input_dim=4, hidden_sizes=config.hidden_sizes, num_classes=3)
    trainer = PyTorchTrainer(model=model, learning_rate=config.learning_rate, epochs=config.epochs, batch_size=config.batch_size)
    evaluator = ModelEvaluator()
    repo = LocalModelRepository()
    tracker = ConsoleTracker()
    return TrainingPipeline(loader, model, trainer, evaluator, repo, tracker)


