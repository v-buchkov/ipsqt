from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipsqt.strategies.optimization_data import TrainingData

import numpy as np
import pandas as pd

from ipsqt.strategies.predicted.base_predicted_strategy import BasePredictedStrategy
from ipsqt.prediction.base_predictor import BasePredictor


class MomentumReversalStrategy(BasePredictedStrategy):
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

        self.target_name = None

    def classify_momentum_reversal(self, row: pd.Series) -> int:
        if np.sign(row[self.target_name]) == np.sign(row["prev_ret"]):
            return 1  # Momentum regime
        else:
            return 0  # Reversal regime

    def construct_target(self, training_data: TrainingData) -> pd.Series:
        self.predictor.model_config.n_classes = 2

        ret = training_data.targets.copy()
        self.target_name = ret.columns[0]
        ret["prev_ret"] = ret.shift(1)
        target = ret.apply(self.classify_momentum_reversal, axis=1)

        return target

    def pred_to_weights(self, predictions: pd.DataFrame) -> pd.Series:
        last_ret = self.seen_training_data.targets.iloc[-1, 0]
        last_ret_sign = np.sign(last_ret)
        weights = predictions.iloc[:, 0].apply(lambda x: 1 if x == 1 else -1)
        weights = weights * last_ret_sign

        return weights
