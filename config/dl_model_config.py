from __future__ import annotations

from typing import Type
from dataclasses import dataclass

import torch
import torch.nn as nn

from ipsqt.config.base_model_config import BaseModelConfig


@dataclass
class DLModelConfig(BaseModelConfig):
    lr: float = 1e-3
    hidden_size: int = 32
    n_layers: int = 2

    dropout: float = 0.0

    n_epochs: int = 10

    optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD
    scheduler: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.CosineAnnealingLR

    weights_decay: float = 0.0

    batch_size: int = 64

    loss: nn.Module = nn.MSELoss()

    clip_grad_norm: float | None = None
