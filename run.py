from __future__ import annotations

import pandas as pd
from ipsqt.config.trading_config import TradingConfig
from ipsqt.backtest.assessor import StrategyStatistics
from ipsqt.runner import Runner
from ipsqt.features.preprocessor import Preprocessor
from config.experiment_config import ExperimentConfig
from config.dl_model_config import DLModelConfig
from ipsqt.prediction.dl.dl_predictor import DLPredictor

from ipsqt.strategies.predicted.binary_position_strategy import BinaryPositionStrategy
from ipsqt.prediction.dl.models.mlp import MLP


REBAL_FREQ = "D"
STRATEGY = BinaryPositionStrategy
MODEL = MLP

SAVE = True


def initialize(
        with_causal_window: bool = True,
        start: str | None = None,
        end: str | None = None,
        rebal_freq: str = REBAL_FREQ,
) -> tuple[Preprocessor, Runner]:
    experiment_config = ExperimentConfig()

    experiment_config.N_LOOKBEHIND_PERIODS = None
    experiment_config.REBALANCE_FREQ = rebal_freq

    if not with_causal_window:
        experiment_config.CAUSAL_WINDOW_SIZE = None

    if start is not None:
        experiment_config.START_DATE = pd.Timestamp(start)
    if end is not None:
        experiment_config.END_DATE = pd.Timestamp(end)

    preprocessor = Preprocessor()

    trading_config = TradingConfig(
        max_exposure=2,
        min_exposure=-2,
        trading_lag_days=0,
    )

    runner = Runner(
        experiment_config=experiment_config,
        trading_config=trading_config,
        verbose=True,
    )

    return preprocessor, runner


def run_backtest() -> StrategyStatistics:
    print("Running backtest...")
    preprocessor, runner = initialize()

    predictor = DLPredictor(
        model_cls=MODEL,
        model_config=DLModelConfig(),
        n_features=len(runner.available_features),
        verbose=False,
    )

    strategy = STRATEGY(
        predictor=predictor,
    )

    result = runner(
        feature_processor=preprocessor,
        strategy=strategy,
        hedger=None,
    )

    strategy_name = STRATEGY.__class__.__name__
    model_name = MODEL.__class__.__name__

    if SAVE:
        runner.save(f"{strategy_name}_" + model_name + f"_rebal{REBAL_FREQ}")

    runner.plot_cumulative(
        strategy_name=strategy_name,
        include_factors=True,
    )

    runner.plot_turnover()

    runner.plot_outperformance(mkt_only=True)

    return result


if __name__ == "__main__":
    run_result = run_backtest()

    print(run_result)  # noqa: T201
