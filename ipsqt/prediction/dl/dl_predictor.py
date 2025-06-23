from __future__ import annotations

from abc import abstractmethod
from typing import Type

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ipsqt.prediction.base_predictor import BasePredictor
from ipsqt.config.base_model_config import BaseModelConfig


class _DLPredictor(BasePredictor):
    def __init__(self, model_cls: Type[nn.Module], model_config: BaseModelConfig = BaseModelConfig(), verbose: bool = False):
        super().__init__(model_config=model_config)

        self.model_config = model_config

        torch.random.manual_seed(model_config.random_seed)
        self.model = model_cls(**model_config.dict())

        self.model = self.model.to(model_config.device)
        self.device = model_config.device
        self.verbose = verbose

        self.optimizer = model_config.optimizer(self.model.parameters(), lr=model_config.lr, weight_decay=model_config.weights_decay)
        self.scheduler = model_config.scheduler(self.optimizer, T_max=model_config.n_epochs)
        self.criterion = model_config.loss

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        features = torch.Tensor(X.to_numpy())
        targets = torch.Tensor(y.to_numpy())

        train_set = TensorDataset(features, targets)
        train_loader = DataLoader(
            train_set,
            batch_size=self.model_config.batch_size,
            shuffle=False, # time series training
            pin_memory=False, # due to mps training
            drop_last=False,
        )

        self._train_model(train_loader)

    def _train_model(self, train_loader: DataLoader) -> None:
        iter = range(self.model_config.n_epochs)
        if self.verbose:
            iter = tqdm(iter)

        for _ in (pbar := iter):
            train_loss = 0.0
            self.model.train()
            for features, labels in train_loader:
                self.optimizer.zero_grad()

                features = features.to(self.device)
                labels = labels.to(self.device)

                pred = self.model(features)

                loss = self.criterion(pred, labels)

                loss.backward()
                self.optimizer.step()

                if self.model_config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.model_config.clip_grad_norm)

                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            self.scheduler.step()
            if self.verbose:
                pbar.set_description(f"Loss: {train_loss:.4f}")

    @abstractmethod
    def _predict_model(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class DLRegressor(_DLPredictor):
    def __init__(self, model_cls: Type[nn.Module], model_config: BaseModelConfig = BaseModelConfig(), verbose: bool = False):
        super().__init__(
            model_cls=model_cls,
            model_config=model_config,
            verbose=verbose,
        )

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        feat_torch = torch.Tensor(X.to_numpy()).to(self.device)
        return self.model(feat_torch).detach().cpu().numpy()


class DLClassifier(_DLPredictor):
    def __init__(self, model_cls: Type[nn.Module], model_config: BaseModelConfig = BaseModelConfig(), verbose: bool = False):
        super().__init__(
            model_cls=model_cls,
            model_config=model_config,
            verbose=verbose,
        )

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        feat_torch = torch.Tensor(X.to_numpy()).to(self.device)
        pred = self.model(feat_torch).detach()
        pred = torch.argmax(pred, dim=1)
        return pred.cpu().numpy()
