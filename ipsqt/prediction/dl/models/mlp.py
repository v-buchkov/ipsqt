import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(self, sizes: list[int], dropout: float = 0.0):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        # for name, param in self.named_parameters():
        #     if "weight_hh" in name:
        #         nn.init.orthogonal_(param)  # Orthogonal initialization
        #     elif "weight_ih" in name:
        #         nn.init.xavier_uniform_(param)  # Xavier initialization
        #     elif "weight" in name:
        #         nn.init.uniform_(param)
        #     elif "bias" in name:
        #         nn.init.zeros_(param)

    def forward(self, x):
        return self.layers(x)


class MLPRegressor(nn.Module):
    def __init__(
        self, hidden_size: int, n_features: int, n_layers: int, *args, **kwargs
    ):
        super().__init__()

        self.model = _MLP([n_features] + ([hidden_size] * n_layers) + [1])

    def forward(self, x: torch.Tensor):
        return self.model(x)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model = _MLP([n_features] + ([hidden_size] * n_layers) + [n_classes])

    def forward(self, x: torch.Tensor):
        return self.model(x)
