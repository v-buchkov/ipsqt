# Semester Paper // Investment Processes - Selected Quantitative Tools, University of Zürich, Spring 2025
_Viacheslav Buchkov_\
University of Zürich

**Abstract**: The paper explores in details the deep learning modeling in market risk premium prediction task. We find that the MLP predictor produces stronger and more robust results than the stronger models. We train the model to recognize the momentum versus reversal patterns. We find the improvement in performance, based on the likelihood steepness, which might be explored further in details.

**Keywords**: Virtue of Complexity; Multi-Layer Perceptron; Market Risk Premium; Bayesian Uncertainty

## Install

```
git clone https://github.com/v-buchkov/ipsqt.git
```

## How To Create A New Predictor

```
from __future__ import annotations

from typing import Type

import numpy as np
import pandas as pd
import torch
from torch import nn

from ipsqt.prediction.dl.dl_predictor import _DLPredictor
from ipsqt.config.base_model_config import BaseModelConfig

class NewDLPredictor(_DLPredictor):
    def __init__(
        self,
        model_cls: Type[nn.Module],
        model_config: BaseModelConfig = BaseModelConfig(),
        verbose: bool = False,
        track_grad_norm: bool = False,
    ):
        super().__init__(
            model_cls=model_cls,
            model_config=model_config,
            verbose=verbose,
            track_grad_norm=track_grad_norm,
        )

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        feat_torch = torch.Tensor(X.to_numpy()).to(self.device)
        model_output = self.model(feat_torch).detach().cpu().numpy()
        # Specify the model prediction logic
        return ...
```

## How To Create A New Prediction-Based Strategy

```
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipsqt.strategies.optimization_data import TrainingData

import pandas as pd

from ipsqt.strategies.predicted.base_predicted_strategy import BasePredictedStrategy
from ipsqt.prediction.base_predictor import BasePredictor


class NewStrategy(BasePredictedStrategy):
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

    def construct_target(self, training_data: TrainingData) -> pd.Series:
        # Construct the machine learning target variable
        return ...

    def pred_to_weights(self, predictions: pd.DataFrame) -> pd.Series:
        # Specify the transformation from predictions to the weights taken
        return ...
```

## How To Run A Backtest

```
from config.dl_model_config import DLModelConfig

from ipsqt.prediction.dl.dl_predictor import DLClassifier
from ipsqt.prediction.dl.models.mlp import MLPClassifier
from ipsqt.strategies.predicted.momentum_reversal_uncert_strategy import MomentumReversalUncertStrategy

from run import initialize

REBAL_FREQ = "ME"
STRATEGY = MomentumReversalUncertStrategy
MODEL = MLPClassifier
RETRAIN_NUM_DAYS = 21

SAVE = True

preprocessor, runner = initialize()

model_config = DLModelConfig()
model_config.n_features = len(runner.available_features)
model_config.n_classes = 2

predictor = DLClassifier(
    model_cls=MODEL,
    model_config=model_config,
    verbose=False,
)

strategy = STRATEGY(
    predictor=predictor,
    retrain_num_days=RETRAIN_NUM_DAYS,
)

result = runner(
    feature_processor=preprocessor,
    strategy=strategy,
    hedger=None,
)
print(result)
```
