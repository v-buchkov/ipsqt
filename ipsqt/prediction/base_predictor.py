from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ipsqt.config.base_model_config import BaseModelConfig


class BasePredictor(ABC):
    def __init__(self, model_config: BaseModelConfig = BaseModelConfig()) -> None:
        super().__init__()

        self.model_config = model_config

        self.feat_scaler = self.model_config.feature_scaler() if self.model_config.feature_scaler is not None else None
        self.target_scaler = self.model_config.target_scaler() if self.model_config.target_scaler is not None else None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.feat_scaler is not None:
            feat_transf = self.feat_scaler.fit_transform(X)
            X = pd.DataFrame(feat_transf, index=X.index, columns=X.columns)

        if self.target_scaler is not None:
            target_transf = self.target_scaler.fit_transform(y)
            y = pd.Series(target_transf, index=y.index, name=y.name)

        self._fit_model(X=X, y=y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feat_scaler is not None:
            feat_transf = self.feat_scaler.transform(X)
            X = pd.DataFrame(feat_transf, index=X.index, columns=X.columns)

        predictions = self._predict_model(X=X)

        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)

        return predictions

    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
