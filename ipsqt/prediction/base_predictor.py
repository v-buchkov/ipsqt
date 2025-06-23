from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
