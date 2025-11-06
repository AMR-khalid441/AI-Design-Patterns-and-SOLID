"""God-class trainer (SRP violation) — minimal runnable implementation.

This class intentionally centralizes configuration, data IO, model construction,
training, evaluation, logging, and saving in one place.
"""

from pathlib import Path
import logging
import random
import yaml
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


logging.basicConfig(level=logging.INFO, format="[violation] %(message)s")
log = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str | None = None, overrides: dict | None = None) -> None:
        self.config_path = config_path
        # Defaults; optionally overridden by YAML
        self.cfg = {
            "learning_rate": 0.001,
            "epochs": 5,
            "hidden_sizes": [16, 8],
            "batch_size": 16,
            "test_size": 0.2,
            "random_state": 42,
            "outputs_dir": str((Path(__file__).resolve().parents[1] / "outputs").as_posix()),
            "seed": 42,
            "use_stratify": True,
        }
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    self.cfg.update(loaded)
        if overrides:
            self.cfg.update(overrides)
        self._validate_cfg()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: nn.Module | None = None
        self.train_x = self.train_y = self.valid_x = self.valid_y = None

    def _validate_cfg(self) -> None:
        required = [
            "learning_rate",
            "epochs",
            "hidden_sizes",
            "batch_size",
            "test_size",
            "random_state",
            "outputs_dir",
            "seed",
        ]
        missing = [k for k in required if k not in self.cfg]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def load_data(self) -> None:  # loads CSV from shared fixtures, splits
        csv_path = Path(__file__).resolve().parents[2] / "shared" / "fixtures" / "sample_data.csv"
        df = pd.read_csv(csv_path)
        X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.astype(np.float32)
        y = df["species"].astype("category").cat.codes.values.astype(np.int64)
        # decide if we can stratify given tiny CSV
        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min() if counts.size > 0 else 0
        n = y.shape[0]
        k = unique.size
        can_stratify = min_count >= 2 and (k * 2) <= n
        want_stratify = bool(self.cfg.get("use_stratify", True))
        stratify_arg = y if (want_stratify and can_stratify) else None
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.cfg["test_size"],
            random_state=self.cfg["random_state"],
            stratify=stratify_arg,
            shuffle=True,
        )
        self.train_x = torch.from_numpy(X_train)
        self.train_y = torch.from_numpy(y_train)
        self.valid_x = torch.from_numpy(X_valid)
        self.valid_y = torch.from_numpy(y_valid)

    def build_model(self) -> None:  # constructs tiny MLP
        input_dim = int(self.train_x.shape[1])
        num_classes = int(self.train_y.max().item()) + 1
        layers: list[nn.Module] = []
        prev = input_dim
        for h in self.cfg["hidden_sizes"]:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        layers.append(nn.Linear(prev, num_classes))
        self.model = nn.Sequential(*layers).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.cfg["learning_rate"]))

    def train(self) -> None:  # simple SGD mini-batch loop
        self.model.train()
        x = self.train_x.to(self.device)
        y = self.train_y.to(self.device)
        epochs = int(self.cfg["epochs"]) 
        bs = int(self.cfg["batch_size"]) 
        n = x.size(0)
        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                xb, yb = x[idx], y[idx]
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            log.info(f"[epoch {epoch}] loss={epoch_loss / n:.4f}")

    def evaluate(self) -> dict:  # metrics
        self.model.eval()
        with torch.no_grad():
            x = self.valid_x.to(self.device)
            y = self.valid_y.to(self.device)
            logits = self.model(x)
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
        metrics = {"accuracy": acc}
        log.info(f"metrics: {metrics}")
        return metrics

    def log_metrics(self, metrics: dict) -> None:  # print/logging
        log.info(f"logging metrics: {metrics}")

    def save_model(self, metadata: dict) -> str:  # persistence
        out_dir = Path(self.cfg["outputs_dir"]) 
        out_dir.mkdir(parents=True, exist_ok=True)
        uri = out_dir / "model.pt"
        torch.save(self.model.state_dict(), uri)
        log.info(f"saved model to: {uri}")
        return str(uri)

    def run(self) -> None:
        # set seeds for reproducibility
        random.seed(int(self.cfg["seed"]))
        torch.manual_seed(int(self.cfg["seed"]))
        np.random.seed(int(self.cfg["seed"]))

        log.info("starting run...")
        self.load_data()
        self.build_model()
        self.train()
        metrics = self.evaluate()
        self.log_metrics(metrics)
        self.save_model({"metrics": metrics})
        log.info("done.")


