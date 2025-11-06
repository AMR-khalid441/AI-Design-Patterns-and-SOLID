"""Simple MLP model."""

from torch import nn


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for h in hidden_sizes:
            h = int(h)
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


