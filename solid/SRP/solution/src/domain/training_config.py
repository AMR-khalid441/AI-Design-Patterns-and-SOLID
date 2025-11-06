"""Training configuration."""

from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    learning_rate: float
    epochs: int
    hidden_sizes: List[int]
    batch_size: int
    seed: int
    test_size: float
    random_state: int
    use_stratify: bool


