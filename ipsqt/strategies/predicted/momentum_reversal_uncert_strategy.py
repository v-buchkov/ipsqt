from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipsqt.strategies.optimization_data import TrainingData, PredictionData

import numpy as np
import pandas as pd

from ipsqt.prediction.base_predictor import BasePredictor
from ipsqt.strategies.predicted.momentum_reversal_strategy import MomentumReversalStrategy


class MomentumReversalUncertStrategy(MomentumReversalStrategy):
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

        self.uncerts = []

    def _get_weights(
        self, prediction_data: PredictionData, weights_: pd.DataFrame
    ) -> pd.DataFrame:
        feat = prediction_data.features

        predictions = self.predictor.predict(feat)
        uncert = self.predictor.uncertainty_estimate

        scaler = np.mean(self.uncerts) / (uncert + 1e-9) if len(self.uncerts) > 0 else 1
        scaler = np.clip(scaler, 0, 2)

        self.uncerts.append(uncert)

        predictions = pd.DataFrame(
            predictions, columns=self.available_assets, index=feat.index
        )
        self._predictions.append([feat.index[-1], predictions.to_numpy().item()])

        weights_.loc[:, self.available_assets] = self.pred_to_weights(predictions) * scaler
        return weights_
