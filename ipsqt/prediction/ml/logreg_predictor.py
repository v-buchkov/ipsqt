from __future__ import annotations

import pandas as pd

from sklearn.linear_model import LogisticRegression

from ipsqt.prediction.base_predictor import BasePredictor
from ipsqt.config.base_model_config import BaseModelConfig


class LogRegPredictor(BasePredictor):
    def __init__(self, model_config: BaseModelConfig = BaseModelConfig()):
        super().__init__(model_config=model_config)

        self.init_model()

    def init_model(self) -> None:
        self.model = LogisticRegression(
            random_state=self.model_config.random_seed,
        )

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def _predict_model(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X)
