from __future__ import annotations

import pandas as pd

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from ipsqt.prediction.base_predictor import BasePredictor
from ipsqt.config.base_model_config import BaseModelConfig


class GPCPredictor(BasePredictor):
    def __init__(self, kernel=RBF(), model_config: BaseModelConfig = BaseModelConfig()):
        super().__init__(model_config=model_config)

        self.kernel = kernel

        self.init_model()

    def init_model(self) -> None:
        self.model = GaussianProcessClassifier(
            kernel=self.kernel,
            n_restarts_optimizer=3,
            random_state=self.model_config.random_seed,
        )

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def _predict_model(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)
