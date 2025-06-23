from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipsqt.strategies.optimization_data import TrainingData

import numpy as np
import pandas as pd

from ipsqt.strategies.predicted.base_predicted_strategy import BasePredictedStrategy
from ipsqt.prediction.base_predictor import BasePredictor
from ipsqt.config.base_model_config import BaseModelConfig


class BinaryPositionStrategy(BasePredictedStrategy):
    def __init__(
        self,
        predictor: BasePredictor,
        model_config: BaseModelConfig,
        window_size: int | None = None,
    ) -> None:
        super().__init__(
            predictor=predictor,
            model_config=model_config,
            window_size=window_size,
        )

    def construct_target(self, training_data: TrainingData) -> pd.Series:
        return training_data.simple_excess_returns.apply(
            lambda x: 1 if x > 0 else -1, axis=1
        )

    def pred_to_weights(self, predictions: pd.DataFrame) -> pd.Series:
        return np.sign(predictions)
