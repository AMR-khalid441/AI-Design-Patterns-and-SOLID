"""CSV data loader."""

from pathlib import Path
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


class CsvLoader:
    def __init__(self, seed: int, test_size: float, random_state: int, use_stratify: bool) -> None:
        self.seed = int(seed)
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.use_stratify = bool(use_stratify)

    def train_valid(self):
        csv_path = Path(__file__).resolve().parents[4] / "shared" / "fixtures" / "sample_data.csv"
        df = pd.read_csv(csv_path)

        X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.astype(np.float32)
        y = df["species"].astype("category").cat.codes.values.astype(np.int64)

        unique, counts = np.unique(y, return_counts=True)
        min_count = counts.min() if counts.size > 0 else 0
        n = y.shape[0]
        k = unique.size
        can_stratify = min_count >= 2 and (k * 2) <= n
        stratify_arg = y if (self.use_stratify and can_stratify) else None
        if stratify_arg is None:
            log.info("CsvLoader: non-stratified split due to tiny class counts or disabled stratify")

        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_arg,
            shuffle=True,
        )

        train_x = torch.from_numpy(X_train)
        train_y = torch.from_numpy(y_train)
        valid_x = torch.from_numpy(X_valid)
        valid_y = torch.from_numpy(y_valid)
        return (train_x, train_y), (valid_x, valid_y)


