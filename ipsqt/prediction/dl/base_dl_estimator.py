from __future__ import annotations

from abc import abstractmethod
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ipsqt.strategies.optimization_data import PredictionData, TrainingData
from ipsqt.cov_estimators.base_cov_estimator import BaseCovEstimator
from ipsqt.cov_estimators.shrinkage.rp_cov_estimator import RiskfolioCovEstimator
from ipsqt.cov_estimators.shrinkage.qis import QISCovEstimator


class ShrinkageType(Enum):
    LINEAR = "linear"
    QIS = "qis"


class BaseRLCovEstimator(BaseCovEstimator):
    def __init__(
        self, shrinkage_type: str = "linear", window_size: int | None = None
    ) -> None:
        super().__init__()

        self.shrinkage_type = ShrinkageType(shrinkage_type)
        self.window_size = window_size

        self.feat_scaler = StandardScaler()

        if self.shrinkage_type == ShrinkageType.QIS:
            self.shrinkage = QISCovEstimator(
                shrinkage=1.0,  # Starting shrinkage, will be updated during fit
            )
        elif self.shrinkage_type == ShrinkageType.LINEAR:
            self.shrinkage = RiskfolioCovEstimator(
                estimator_type="shrunk",
                alpha=0.1,  # Starting alpha, will be updated during fit
            )
        else:
            msg = f"Unknown shrinkage type: {self.shrinkage_type}"
            raise NotImplementedError(msg)

        self._seen_training_data = None

        self._predictions = []
        self.trained_with_features = False

    @abstractmethod
    def _fit_shrinkage(
        self, features: pd.DataFrame, shrinkage_target: pd.Series
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def _predict_shrinkage(self, features: pd.DataFrame) -> float:
        raise NotImplementedError

    def _fit(self, training_data: TrainingData) -> None:

        self.shrinkage.available_assets = self.available_assets
        self._fit_shrinkage(features=feat, shrinkage_target=target)

    def _predict(self, prediction_data: PredictionData) -> pd.DataFrame:
        feat = prediction_data.features
        if not self.predictions.empty and self.trained_with_features:
            feat["prediction"] = self.predictions.iloc[-1].item()
        feat_transformed = self.feat_scaler.transform(feat)
        feat = pd.DataFrame(feat_transformed, index=feat.index, columns=feat.columns)

        pred_shrinkage = self._predict_shrinkage(feat)

        pred_shrinkage = (
            np.clip(pred_shrinkage, 0, 1)
            if self.shrinkage_type == ShrinkageType.LINEAR
            else pred_shrinkage
        )

        if self.shrinkage_type == ShrinkageType.LINEAR:
            self.shrinkage.alpha = pred_shrinkage
        elif self.shrinkage_type == ShrinkageType.QIS:
            self.shrinkage.shrinkage = pred_shrinkage
        else:
            msg = f"Unknown shrinkage type: {self.shrinkage_type}"
            raise NotImplementedError(msg)

        self._predictions.append([feat.index[-1], pred_shrinkage])

        self.shrinkage.fit(training_data=self._seen_training_data)

        self._seen_training_data = None

        return self.shrinkage.predict(prediction_data=prediction_data)
