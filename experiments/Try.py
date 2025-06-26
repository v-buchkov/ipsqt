#%%
import numpy as np
import pandas as pd
import torch
from torch import nn

from config.experiment_config import ExperimentConfig

config = ExperimentConfig()
#%%
market_data = pd.read_excel(config.PATH_INPUT / config.INPUT_DATA_FILENAME)
market_data = market_data.rename(columns={"Date": "date"})
market_data["date"] = pd.to_datetime(market_data["date"])
market_data = market_data.set_index("date")
#%%
X = np.log(market_data[["CAPE"]].shift(1))
#%%
def classify_momentum_reversal(row: pd.Series) -> int:
    if np.sign(row["_MKT"]) == np.sign(row["prev_ret"]):
        return 1  # Momentum regime
    else:
        return 0  # Reversal regime

def construct_target(ret) -> pd.Series:
    ret = ret.copy()
    ret["prev_ret"] = ret.shift(1)
    target = ret.apply(classify_momentum_reversal, axis=1)

    return target

# y = construct_target(data[["_MKT"]])
y = market_data["_MKT"].pct_change(52)
y = y.dropna()
X = X.loc[y.index[0]:]
#%%
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

ss = StandardScaler()
X = ss.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)
train_set = TensorDataset(X, y)

train_loader = DataLoader(train_set, batch_size=32, shuffle=False, pin_memory=False, drop_last=False)
#%%
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
#%%
from IPython.display import clear_output
import matplotlib.pyplot as plt


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
#%%
from ipsqt.prediction.dl.models.mlp import MLPClassifier, MLPRegressor

# model = MLPClassifier(
#     n_features=X.shape[1],
#     n_classes=2,
#     n_layers=1,
#     hidden_size=8,
# )
model = MLPRegressor(
    n_features=X.shape[1],
    n_layers=1,
    hidden_size=8,
)
model = model.to(device)

N_EPOCHS = 20

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
#%%
from tqdm import tqdm

grad_norms = []
train_losses = []
for epoch in (pbar := tqdm(range(N_EPOCHS))):
    train_loss = 0.0
    pred_path = []
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()

        features = features.to(device)
        labels = labels.to(device)

        pred = model(features)

        loss = criterion(pred, labels)
        # true_balance = labels.mean().item()
        # acc = (pred.detach().argmax(axis=1) == labels).to(torch.float32).mean().item()

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.norm().item()
                total_norm += param_norm**2
        total_norm = total_norm**0.5
        grad_norms.append(total_norm)

        loss.backward()
        optimizer.step()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)
    # scheduler.step()
    # pbar.set_description(f"Loss: {train_loss:.4f}, Accuracy: {acc:.4f}, True Balance: {true_balance:.4f}")
    pbar.set_description(f"Loss: {train_loss:.4f}")
    plot_losses(train_losses, grad_norms)
#%%
model.eval()
preds = []
for features, labels in train_loader:
    features = features.to(device)
    labels = labels.to(device)

    pred = model(features).detach().cpu().numpy()
    preds.append(pred)
#%%
from sklearn.metrics import r2_score

print(r2_score(y.cpu(), np.concat(preds).reshape(-1)))
#%%
