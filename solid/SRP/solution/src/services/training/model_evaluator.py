"""Model evaluator."""

import torch
from torch import nn


class ModelEvaluator:
    def __init__(self) -> None:
        pass

    def evaluate(self, model: nn.Module, valid_xy) -> dict:
        model.eval()
        valid_x, valid_y = valid_xy
        with torch.no_grad():
            logits = model(valid_x)
            preds = logits.argmax(dim=1)
            acc = (preds == valid_y).float().mean().item()
        return {"accuracy": acc}


