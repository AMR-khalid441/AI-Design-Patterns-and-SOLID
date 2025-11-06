"""PyTorch trainer."""

import logging
import torch
from torch import nn, optim

log = logging.getLogger(__name__)


class PyTorchTrainer:
    def __init__(self, model: nn.Module, learning_rate: float, epochs: int, batch_size: int) -> None:
        self.model = model
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, train_xy) -> None:
        train_x, train_y = train_xy
        self.model.train()
        x = train_x
        y = train_y
        n = x.size(0)
        for epoch in range(1, self.epochs + 1):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = x[idx], y[idx]
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            log.info(f"[solution][epoch {epoch}] loss={epoch_loss / n:.4f}")


