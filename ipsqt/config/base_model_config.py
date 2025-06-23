from __future__ import annotations

from typing import Type
from dataclasses import dataclass, asdict

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
import torch


@dataclass
class BaseModelConfig:
    n_features: int | None = None
    random_seed: int = 12

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    feature_scaler: Type[BaseEstimator] | None = StandardScaler
    target_scaler: Type[BaseEstimator] | None  = None

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
