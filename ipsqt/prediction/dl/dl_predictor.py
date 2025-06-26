from __future__ import annotations

from abc import abstractmethod
from typing import Type
from IPython.display import clear_output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ipsqt.prediction.base_predictor import BasePredictor
from ipsqt.config.base_model_config import BaseModelConfig
from ipsqt.prediction.dl.models.lstm import LSTMClassifier


class _DLPredictor(BasePredictor):
    def __init__(
        self,
        model_cls: Type[nn.Module],
        model_config: BaseModelConfig = BaseModelConfig(),
        verbose: bool = False,
        track_grad_norm: bool = False,
    ):
        super().__init__(model_config=model_config)

        self.model_cls = model_cls
        self.model_config = model_config

        self.device = model_config.device
        self.verbose = verbose
        self.track_grad_norm = track_grad_norm

        self.init_model()

        self.optimizer = model_config.optimizer(
            self.model.parameters(),
            lr=model_config.lr,
            weight_decay=model_config.weights_decay,
        )
        self.scheduler = model_config.scheduler(
            self.optimizer, T_max=model_config.n_epochs
        ) if model_config.scheduler is not None else None
        self.criterion = model_config.loss

    def init_model(self) -> None:
        torch.random.manual_seed(self.model_config.random_seed)
        self.model = self.model_cls(**self.model_config.dict())
        self.model = self.model.to(self.device)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        features = torch.Tensor(X.to_numpy())
        targets = torch.Tensor(y.to_numpy())

        train_set = TensorDataset(features, targets)
        train_loader = DataLoader(
            train_set,
            batch_size=self.model_config.batch_size,
            shuffle=False,  # time series training
            pin_memory=False,  # due to mps training
            drop_last=False,
        )

        self._train_model(train_loader)

    def _train_model(self, train_loader: DataLoader) -> None:
        iter = range(self.model_config.n_epochs)
        if self.verbose:
            iter = tqdm(iter)

        grad_norms = []
        train_losses = []
        self.accurracies = []
        self.true_balances = []
        for _ in (pbar := iter):
            train_loss = 0.0
            self.model.train()

            if isinstance(self.model, LSTMClassifier):
                h_t, c_t = None, None

            for features, labels in train_loader:
                self.optimizer.zero_grad()

                features = features.to(self.device)
                labels = labels.to(self.device)

                if isinstance(self.model, LSTMClassifier):
                    pred, (h_t, c_t) = self.model(features, h_t, c_t)
                    h_t, c_t = h_t.detach(), c_t.detach()
                else:
                    pred = self.model(features)

                loss = self.criterion(pred, labels)
                true_balance = labels.mean().item()
                acc = (pred.detach().argmax(axis=1) == labels).to(torch.float32).mean().item()

                loss.backward()

                if self.track_grad_norm:
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.norm().item()
                            total_norm += param_norm**2
                    total_norm = total_norm**0.5
                    grad_norms.append(total_norm)

                self.optimizer.step()

                if self.model_config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.model_config.clip_grad_norm,
                    )

                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            self.accurracies.append(acc)
            self.true_balances.append(true_balance)
            if self.scheduler is not None:
                self.scheduler.step()
            if self.verbose:
                pbar.set_description(f"Loss: {train_loss:.4f} Accuracy: {acc:.4f}")
                self.plot_losses(train_losses, grad_norms if self.track_grad_norm else None)

    @staticmethod
    def plot_losses(
        train_losses: list[float],
        grad_norms: list[float] | None = None,
    ):
        clear_output()
        n_cols = 2 if grad_norms is not None else 1
        fig, axs = plt.subplots(1, n_cols, figsize=(13, 4))

        if n_cols == 1:
            axs = [axs]

        axs[0].plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
        axs[0].set_ylabel("Loss")

        if grad_norms is not None:
            axs[1].plot(
                range(1, len(grad_norms) + 1),
                grad_norms,
                label="grad_norm",
            )
            axs[1].set_ylabel("Gradient Norm Over Training")

        for ax in axs:
            ax.set_xlabel("epoch")
            ax.legend()

        plt.show()

    @abstractmethod
    def _predict_model(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class DLRegressor(_DLPredictor):
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
        return self.model(feat_torch).detach().cpu().numpy()


class DLClassifier(_DLPredictor):
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
        if isinstance(self.model, LSTMClassifier):
            pred, _ = self.model(feat_torch)
            pred = pred.detach()
        else:
            pred = self.model(feat_torch).detach()

        pred = torch.argmax(pred, dim=1)
        return pred.cpu().numpy()
