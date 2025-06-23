from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
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

        self.hidden_size = hidden_size
        self.n_features = n_features
        self.n_layers = n_layers

        self.model = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bias=False,
        )

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes),
        )

        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)  # Orthogonal initialization
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)  # Xavier initialization

    def forward(
        self,
        x: torch.Tensor,
        h_t: torch.Tensor | None = None,
        c_t: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        device = x.device

        x = x.unsqueeze(1)

        if h_t is None:
            h_t = torch.zeros(
                self.n_layers,
                x.shape[0],
                self.hidden_size,
                dtype=torch.float32,
                requires_grad=True,
            ).to(device)

        if c_t is None:
            c_t = torch.zeros(
                self.n_layers,
                x.shape[0],
                self.hidden_size,
                dtype=torch.float32,
                requires_grad=True,
            ).to(device)

        if h_t.shape[1] != x.shape[0]:
            h_t = h_t[:, -x.shape[0] :, :]
            c_t = c_t[:, -x.shape[0] :, :]

        out, (h_t, c_t) = self.model(x, (h_t, c_t))
        out = out[:, -1, :]

        return self.final_layer(out), (h_t, c_t)
