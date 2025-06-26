from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipsqt.strategies.optimization_data import TrainingData

import numpy as np
import pandas as pd

from ipsqt.strategies.predicted.base_predicted_strategy import BasePredictedStrategy
from ipsqt.prediction.base_predictor import BasePredictor


class RegressionStrategy(BasePredictedStrategy):
    def __init__(
        self,
        predictor: BasePredictor,
        window_size: int | None = None,
        retrain_num_days: int | None = None,
    ) -> None:
        super().__init__(
            predictor=predictor,
            window_size=window_size,
            retrain_num_days=retrain_num_days,
        )

    def construct_target(self, training_data: TrainingData) -> pd.Series:
        return training_data.targets

    def pred_to_weights(self, predictions: pd.DataFrame) -> pd.Series:
        return np.sign(predictions)
