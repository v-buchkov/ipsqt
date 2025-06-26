#%%
from __future__ import annotations

#%%
from config.dl_model_config import DLModelConfig

from ipsqt.prediction.dl.dl_predictor import DLClassifier as Predictor
from ipsqt.prediction.dl.models.mlp import MLPClassifier as Model
from ipsqt.strategies.predicted.momentum_reversal_strategy import (
    MomentumReversalStrategy as Strategy,
)

from run import initialize
#%%
REBAL_FREQ = "ME"
RETRAIN_NUM_DAYS = 21
SAVE = True
#%%
preprocessor, runner = initialize()

model_config = DLModelConfig()
model_config.n_features = len(runner.available_features)
print(runner.available_features)
model_config.n_classes = 2

model_config.n_epochs = 10

predictor = Predictor(
    model_cls=Model,
    model_config=model_config,
    verbose=False,
)

strategy = Strategy(
    predictor=predictor,
    retrain_num_days=RETRAIN_NUM_DAYS,
)

strategy_name = strategy.__class__.__name__
model_name = predictor.model.__class__.__name__
#%%
result = runner(
    feature_processor=preprocessor,
    strategy=strategy,
    hedger=None,
)
#%%
print(result)
#%%
runner.plot_cumulative(
    strategy_name=strategy_name,
    include_factors=True,
)
#%%
runner.plot_turnover()
#%%
runner.plot_outperformance(mkt_only=True)
#%%
if SAVE:
    runner.save(f"{strategy_name}_" + model_name + f"_rebal{REBAL_FREQ}")
#%%
runner.strategy_weights.plot();
#%%
strategy_total_r = runner.strategy_total_r
#%%
import quantstats as qs

qs.extend_pandas()
qs.plots.snapshot(strategy_total_r.iloc[:, 0], title="MLP", show=True)
#%%
qs.plots.drawdowns_periods(
    strategy_total_r.iloc[:, 0],
    title="MLP",
    figsize=(12, 8),
    show=True,
)
#%%
qs.plots.monthly_heatmap(
    strategy_total_r.iloc[:, 0],
    benchmark=runner.factors.add(runner.rf, axis=0),
    figsize=(12, 8),
    show=True,
)