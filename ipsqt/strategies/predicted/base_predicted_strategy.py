from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipsqt.strategies.optimization_data import PredictionData, TrainingData

import pandas as pd

from ipsqt.strategies.base_strategy import BaseStrategy
from ipsqt.prediction.base_predictor import BasePredictor
from ipsqt.config.base_model_config import BaseModelConfig


class BasePredictedStrategy(ABC, BaseStrategy):
    PERCENTAGE_VALID_POINTS = 1.0

    def __init__(
        self,
        predictor: BasePredictor,
        model_config: BaseModelConfig,
        window_size: int | None = None,
    ) -> None:
        super().__init__()

        self.predictor = predictor
        self.model_config = model_config
        self.window_size = window_size

        self.feat_scaler = self.model_config.feature_scaler() if self.model_config.feature_scaler is not None else None
        self.target_scaler = self.model_config.target_scaler() if self.model_config.target_scaler is not None else None

    def _filter_stocks(self, training_data: TrainingData) -> TrainingData:
        ret = training_data.simple_excess_returns[self.available_assets]

        start_date = (
            ret.index[-1] - pd.Timedelta(days=self.window_size)
            if self.window_size is not None
            else ret.index[0]
        )
        ret = ret.loc[start_date:]

        n_valid_points = (~ret.isna()).sum(axis=0) / len(ret)
        valid_stocks = list(
            n_valid_points[n_valid_points >= self.PERCENTAGE_VALID_POINTS].index
        )

        self.available_assets = valid_stocks

        training_data.simple_excess_returns = training_data.simple_excess_returns[
            self.available_assets
        ]
        training_data.simple_excess_returns = training_data.simple_excess_returns.loc[
            start_date:
        ]

        training_data.log_excess_returns = (
            training_data.log_excess_returns.loc[start_date:, self.available_assets]
            if training_data.log_excess_returns is not None
            else None
        )
        training_data.factors = training_data.factors.loc[start_date:]

        return training_data

    def _get_prediction_data(self, training_data: TrainingData) -> tuple[pd.DataFrame, pd.Series]:
        feat = training_data.features
        target = self.construct_target(training_data=training_data)

        start_date = (
            target.index[-1] - pd.Timedelta(days=self.window_size)
            if self.window_size is not None
            else target.index[0]
        )
        feat = feat.loc[start_date:]
        target = target.loc[start_date:]

        first_feat_date = feat.dropna(axis=0, how="any").first_valid_index()
        last_feat_date = feat.dropna(axis=0, how="any").last_valid_index()

        first_target_date = target.dropna(axis=0, how="any").first_valid_index()
        last_target_date = target.dropna(axis=0, how="any").last_valid_index()

        first_date = (
            first_feat_date
            if first_feat_date >= first_target_date
            else first_target_date
        )
        last_date = (
            last_feat_date if last_target_date >= last_feat_date else last_target_date
        )

        feat = feat.loc[first_date:last_date]
        target = target.loc[first_date:last_date]

        if self.feat_scaler is not None:
            feat_transf = self.feat_scaler.fit_transform(feat)
            feat = pd.DataFrame(feat_transf, index=feat.index, columns=feat.columns)

        if self.target_scaler is not None:
            target_transf = self.target_scaler.fit_transform(target)
            target = pd.Series(target_transf, index=target.index, name=target.name)

        return feat, target

    def _fit(self, training_data: TrainingData) -> None:
        training_data = self._filter_stocks(training_data=training_data)
        features, targets = self._get_prediction_data(training_data=training_data)

        self.predictor.fit(X=features, y=targets)

    @abstractmethod
    def construct_target(self, training_data: TrainingData) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def pred_to_weights(self, predictions: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def _get_weights(
        self, prediction_data: PredictionData, weights_: pd.DataFrame
    ) -> pd.DataFrame:
        feat = prediction_data.features

        if self.feat_scaler is not None:
            feat_transf = self.feat_scaler.transform(feat)
            feat = pd.DataFrame(feat_transf, index=feat.index, columns=feat.columns)

        predictions = self.predictor.predict(feat)

        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions)

        predictions = pd.DataFrame(
            predictions, columns=self.available_assets, index=predictions.index
        )

        weights_.loc[:, self.available_assets] = self.pred_to_weights(predictions)
        return weights_

    @property
    def predictions(self) -> pd.DataFrame | None:
        if self._predictions is None:
            return None

        pred = pd.DataFrame(self._predictions, columns=["date", "prediction"])
        pred["date"] = pd.to_datetime(pred["date"])
        return pred.set_index("date")
