#%%
from __future__ import annotations

#%%
from config.dl_model_config import DLModelConfig

from ipsqt.prediction.ml.logreg_predictor import LogRegPredictor as Predictor
from ipsqt.strategies.predicted.momentum_reversal_strategy import (
    MomentumReversalStrategy as Strategy,
)

from run import initialize
#%%
REBAL_FREQ = "ME"
RETRAIN_NUM_DAYS = 30
SAVE = True
#%%
preprocessor, runner = initialize()

model_config = DLModelConfig()
model_config.n_features = len(runner.available_features)
model_config.n_classes = 2
model_config.n_epochs = 20

predictor = Predictor(model_config=model_config)

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
runner.strategy_weights.plot()