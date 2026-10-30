"""Models: classical SOH regressors, a small LSTM, and a trend-based RUL estimator."""

import random
from itertools import product

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed Python and NumPy RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def make_random_forest(seed: int = SEED, min_samples_leaf: int = 2, max_features: float = 0.5):
    """Random forest baseline: bagged trees, robust and easy to interpret."""
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=seed,
        n_jobs=1,
    )


def make_ridge(seed: int = SEED, alpha: float = 1.0):
    """Standardised ridge regression: a linear model that can extrapolate beyond training SOH."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=seed))


def _grid(**options) -> list[dict]:
    """Expand {name: [values]} into a list of parameter dicts."""
    return [dict(zip(options, combo)) for combo in product(*options.values())]


MODELS = {"random_forest": make_random_forest, "ridge": make_ridge}
PARAM_GRIDS = {
    "random_forest": _grid(min_samples_leaf=[1, 2, 5, 10], max_features=[0.33, 0.5, 1.0]),
    "ridge": _grid(alpha=[0.01, 0.1, 1.0, 10.0, 100.0]),
}


LSTM_WINDOW = 10


def set_torch_seeds(seed: int = SEED) -> None:
    """Seed everything and force deterministic, single-threaded torch ops."""
    set_seeds(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def make_sequences(X: np.ndarray, window: int = LSTM_WINDOW) -> np.ndarray:
    """Turn one cell's (cycles, features) matrix into (cycles, window, features) windows.

    Window k holds cycles k-window+1 .. k. Early cycles are left-padded with cycle 1,
    so every cycle gets a prediction and no future data is ever used.
    """
    padded = np.vstack([np.repeat(X[:1], window - 1, axis=0), X])
    return np.stack([padded[k:k + window] for k in range(len(X))])


class SOHLSTM(nn.Module):
    """Single-layer LSTM over a window of per-cycle features, predicting SOH at the last cycle."""

    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def train_lstm(X_train, y_train, X_val, y_val, seed: int = SEED, max_epochs: int = 300,
               patience: int = 30, lr: float = 3e-3, batch_size: int = 32):
    """Train with Adam and early stopping on validation RMSE; return (best model, best val RMSE)."""
    set_torch_seeds(seed)
    model = SOHLSTM(X_train.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    Xv, yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)

    best_rmse, best_state, stale = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=gen).split(batch_size):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_rmse = float(torch.sqrt(nn.functional.mse_loss(model(Xv), yv)))
        if val_rmse < best_rmse:
            best_rmse, stale = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_rmse


def predict_lstm(model, X) -> np.ndarray:
    """Predict with a trained LSTM on (n, window, features) input."""
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


RUL_WINDOWS = [20, 30, 50, None]


def estimate_rul(cycles, soh, eol_soh: float = 0.70, window: int | None = 30,
                 min_points: int = 10, max_rul: float = 300.0) -> np.ndarray:
    """Remaining cycles until SOH reaches `eol_soh`, estimated at every cycle.

    At cycle k, fit a straight line to the last `window` SOH values up to k (past only;
    window=None uses all history) and extrapolate to the EOL threshold. NaN until
    `min_points` cycles exist; a flat or rising trend gives `max_rul`. Clipped to [0, max_rul].
    """
    cycles, soh = np.asarray(cycles, float), np.asarray(soh, float)
    rul = np.full(len(cycles), np.nan)
    for k in range(min_points - 1, len(cycles)):
        lo = 0 if window is None else max(0, k - window + 1)
        slope, intercept = np.polyfit(cycles[lo:k + 1], soh[lo:k + 1], 1)
        if slope >= 0:
            rul[k] = max_rul
        else:
            crossing = (eol_soh - intercept) / slope
            rul[k] = np.clip(crossing - cycles[k], 0, max_rul)
    return rul
