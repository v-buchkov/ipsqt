# %%
from __future__ import annotations

# %%
import torch

from config.dl_model_config import DLModelConfig

from ipsqt.prediction.dl.dl_predictor import DLClassifier as Predictor
from ipsqt.prediction.dl.models.lstm import LSTMClassifier as Model
from ipsqt.strategies.predicted.momentum_reversal_strategy import (
    MomentumReversalStrategy as Strategy,
)

from run import initialize

# %%
REBAL_FREQ = "D"
RETRAIN = True
SAVE = True
# %%
preprocessor, runner = initialize()

model_config = DLModelConfig()
model_config.optimizer = torch.optim.AdamW
model_config.n_features = len(runner.available_features)
model_config.n_classes = 2

model_config.n_epochs = 20

predictor = Predictor(
    model_cls=Model,
    model_config=model_config,
    verbose=False,
)

strategy = Strategy(
    predictor=predictor,
    retrain=RETRAIN,
)

strategy_name = strategy.__class__.__name__
model_name = predictor.model.__class__.__name__
# %%
result = runner(
    feature_processor=preprocessor,
    strategy=strategy,
    hedger=None,
)
# %%
result
# %%
runner.plot_cumulative(
    strategy_name=strategy_name,
    include_factors=True,
)
# %%
runner.plot_turnover()
# %%
runner.plot_outperformance(mkt_only=True)
# %%
if SAVE:
    runner.save(f"{strategy_name}_" + model_name + f"_rebal{REBAL_FREQ}")
# %%
runner.strategy_weights.plot()
# %%
(
    (runner.strategy_weights == -1).mean().item(),
    (runner.strategy_weights == 1).mean().item(),
)
