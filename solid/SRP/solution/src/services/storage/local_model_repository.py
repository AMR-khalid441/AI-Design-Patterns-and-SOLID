"""Local filesystem model repository."""

from pathlib import Path
import torch
from torch import nn


class LocalModelRepository:
    def __init__(self) -> None:
        self.base = Path(__file__).resolve().parents[3] / "outputs"

    def save(self, model: nn.Module, metadata: dict) -> str:
        self.base.mkdir(parents=True, exist_ok=True)
        uri = self.base / "model.pt"
        torch.save(model.state_dict(), uri)
        return str(uri)


