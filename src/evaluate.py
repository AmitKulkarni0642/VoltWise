"""Leave-one-cell-out evaluation with validation cells, RUL scoring, and the pipeline CLI.

Run a stage with: python -m src.evaluate {eda,baseline,lstm,rul,all}
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import CELLS, EOL_SOH, ROOT, build_cycle_table
from src.features import FEATURES, build_feature_table
from src.models import (LSTM_WINDOW, MODELS, PARAM_GRIDS, RUL_WINDOWS, SEED, estimate_rul,
                        make_sequences, predict_lstm, set_seeds, train_lstm)
from src.plots import (FIG_DIR, plot_discharge_curves, plot_importance, plot_predictions,
                       plot_rul, plot_soh_curves)

RESULTS_DIR = ROOT / "results"
HEADLINE = "ridge_no_capacity_proxies"

# At a constant 2 A discharge, the time to (nearly) empty is close to a direct capacity
# measurement, so every model is also evaluated without these features.
CAPACITY_PROXIES = ["discharge_time_s", "t_to_3p0v_s", "t_to_3p2v_s", "t_to_temp_max_s"]
FEATURE_SETS = {
    "all_features": FEATURES,
    "no_capacity_proxies": [f for f in FEATURES if f not in CAPACITY_PROXIES],
}


def _rmse_mae_pp(y_true, y_pred) -> tuple[float, float]:
    """RMSE and MAE in SOH percentage points."""
    err = 100 * (np.asarray(y_pred) - np.asarray(y_true))
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err)))


def _add_mean_row(metrics: list[dict]) -> pd.DataFrame:
    """Metrics table with a final row averaging the per-cell scores."""
    df = pd.DataFrame(metrics)
    means = df[["val_rmse_pp", "rmse_pp", "mae_pp"]].mean()
    df.loc[len(df)] = {"test_cell": "mean", **means.to_dict(), "params": ""}
    return df


def make_splits(cells=CELLS) -> pd.DataFrame:
    """Cell-level splits: each fold holds out one test cell; the other three rotate as
    validation (one at a time) for model selection, then all three are used to refit.
    """
    rows = []
    for test_cell in cells:
        dev = [c for c in cells if c != test_cell]
        for val_cell in dev:
            rows.append({"test_cell": test_cell, "val_cell": val_cell,
                         "train_cells": ", ".join(c for c in dev if c != val_cell)})
    return pd.DataFrame(rows)


def select_params(dev: pd.DataFrame, features: list[str], make_model, grid: list[dict]):
    """Pick the params with the lowest mean validation RMSE, rotating each dev cell as validation.

    Returns (best params, their mean validation RMSE in pp).
    """
    scores = []
    for params in grid:
        val_rmse = []
        for val_cell in dev["cell"].unique():
            train, val = dev[dev.cell != val_cell], dev[dev.cell == val_cell]
            model = make_model(SEED, **params).fit(train[features], train["soh"])
            val_rmse.append(_rmse_mae_pp(val["soh"], model.predict(val[features]))[0])
        scores.append(float(np.mean(val_rmse)))
    best = int(np.argmin(scores))
    return grid[best], scores[best]


def leave_one_cell_out(table: pd.DataFrame, features: list[str], make_model, grid: list[dict],
                       cells=CELLS):
    """Nested leave-one-cell-out: select params on validation cells, test once on the unseen cell.

    Returns (metrics, predictions, mean importance or None for models without importances).
    """
    metrics, preds, importances = [], [], []
    for test_cell in cells:
        dev, test = table[table.cell != test_cell], table[table.cell == test_cell]
        params, val_rmse = select_params(dev, features, make_model, grid)

        model = make_model(SEED, **params).fit(dev[features], dev["soh"])
        y_pred = model.predict(test[features])
        rmse, mae = _rmse_mae_pp(test["soh"], y_pred)
        metrics.append({"test_cell": test_cell, "val_rmse_pp": val_rmse,
                        "rmse_pp": rmse, "mae_pp": mae, "params": str(params)})
        preds.append(test[["cell", "cycle", "soh"]].assign(soh_pred=y_pred))
        if hasattr(model, "feature_importances_"):
            importances.append(pd.Series(model.feature_importances_, index=features))

    importance = None
    if importances:
        importance = pd.concat(importances, axis=1).mean(axis=1)
        importance /= importance.sum()
    return _add_mean_row(metrics), pd.concat(preds, ignore_index=True), importance


def run_baseline() -> pd.DataFrame:
    """Evaluate every classical model on every feature set; save results and plots."""
    set_seeds()
    table = build_feature_table()
    RESULTS_DIR.mkdir(exist_ok=True)
    make_splits().to_csv(RESULTS_DIR / "splits.csv", index=False)
    all_metrics = []
    for model_name, make_model in MODELS.items():
        model_label = model_name.replace("_", " ").title()
        for set_name, feats in FEATURE_SETS.items():
            metrics, preds, importance = leave_one_cell_out(table, feats, make_model,
                                                             PARAM_GRIDS[model_name])
            tag, set_label = f"{model_name}_{set_name}", set_name.replace("_", " ")
            preds.to_csv(RESULTS_DIR / f"predictions_{tag}.csv", index=False)
            plot_predictions(preds, metrics, f"{model_label}, leave-one-cell-out ({set_label})",
                             FIG_DIR / f"pred_vs_actual_{tag}.png")
            if importance is not None:
                plot_importance(importance, f"{model_label} feature importance ({set_label})",
                                FIG_DIR / f"feature_importance_{tag}.png")
            all_metrics.append(metrics.assign(model=model_name, feature_set=set_name))
    results = pd.concat(all_metrics, ignore_index=True)
    results.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    return results


def _scaled_sequences(table, cells, features, x_scaler, y_mean, y_std):
    """Scaled LSTM windows and targets, built per cell so windows never cross cells."""
    X, y = [], []
    for c in cells:
        g = table[table.cell == c]
        X.append(make_sequences(x_scaler.transform(g[features])))
        y.append(((g["soh"] - y_mean) / y_std).to_numpy())
    return np.concatenate(X), np.concatenate(y)


def lstm_leave_one_cell_out(table: pd.DataFrame, features: list[str], cells=CELLS):
    """For each test cell, train one LSTM per validation rotation (two train cells, early
    stopping on the validation cell) and average their test predictions.

    Feature and target scaling are fit on the training cells only.
    """
    metrics, preds = [], []
    for test_cell in cells:
        dev = [c for c in cells if c != test_cell]
        test = table[table.cell == test_cell]
        test_preds, val_scores = [], []
        for val_cell in dev:
            train_cells = [c for c in dev if c != val_cell]
            train = table[table.cell.isin(train_cells)]
            scaling = (StandardScaler().fit(train[features]), train["soh"].mean(), train["soh"].std())
            _, y_mean, y_std = scaling

            X_tr, y_tr = _scaled_sequences(table, train_cells, features, *scaling)
            X_val, y_val = _scaled_sequences(table, [val_cell], features, *scaling)
            X_te, _ = _scaled_sequences(table, [test_cell], features, *scaling)
            model, _ = train_lstm(X_tr, y_tr, X_val, y_val, seed=SEED)
            val_pred = predict_lstm(model, X_val) * y_std + y_mean
            val_scores.append(_rmse_mae_pp(y_val * y_std + y_mean, val_pred)[0])
            test_preds.append(predict_lstm(model, X_te) * y_std + y_mean)

        y_pred = np.mean(test_preds, axis=0)
        rmse, mae = _rmse_mae_pp(test["soh"], y_pred)
        metrics.append({"test_cell": test_cell, "val_rmse_pp": float(np.mean(val_scores)),
                        "rmse_pp": rmse, "mae_pp": mae, "params": f"window={LSTM_WINDOW}"})
        preds.append(test[["cell", "cycle", "soh"]].assign(soh_pred=y_pred))
    return _add_mean_row(metrics), pd.concat(preds, ignore_index=True)


def run_lstm() -> pd.DataFrame:
    """Evaluate the LSTM on every feature set; save results and plots."""
    table = build_feature_table()
    RESULTS_DIR.mkdir(exist_ok=True)
    all_metrics = []
    for set_name, feats in FEATURE_SETS.items():
        metrics, preds = lstm_leave_one_cell_out(table, feats)
        tag, set_label = f"lstm_{set_name}", set_name.replace("_", " ")
        preds.to_csv(RESULTS_DIR / f"predictions_{tag}.csv", index=False)
        plot_predictions(preds, metrics, f"LSTM, leave-one-cell-out ({set_label})",
                         FIG_DIR / f"pred_vs_actual_{tag}.png")
        all_metrics.append(metrics.assign(model="lstm", feature_set=set_name))
    results = pd.concat(all_metrics, ignore_index=True)
    results.to_csv(RESULTS_DIR / "lstm_metrics.csv", index=False)
    return results


def _rul_error(rul: pd.DataFrame, cells) -> tuple[float, float, int]:
    """RUL RMSE, MAE (cycles) and count over pre-EOL cycles with an estimate, for `cells`."""
    g = rul[rul.cell.isin(cells) & (rul.rul_true > 0) & rul.rul_pred.notna()]
    err = g["rul_pred"] - g["rul_true"]
    return float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err))), len(g)


def _rul_for_window(df: pd.DataFrame, window) -> pd.DataFrame:
    """True and estimated RUL for every cell in `df`, using one trend window."""
    parts = [g[["cell", "cycle", "eol_cycle"]].assign(
                 rul_true=g["eol_cycle"] - g["cycle"],
                 rul_pred=estimate_rul(g["cycle"], g["soh_pred"], EOL_SOH, window=window))
             for _, g in df.groupby("cell")]
    return pd.concat(parts, ignore_index=True)


def run_rul() -> pd.DataFrame:
    """Estimate RUL from every saved SOH prediction set, plus measured SOH as a reference.

    True RUL = first EOL cycle - current cycle. It is scored only on cells that reach EOL
    (B0007 never does), on cycles before EOL with enough history. The trend window for
    each cell is chosen using only the other EOL cells.
    """
    cycles = build_cycle_table()[["cell", "cycle", "soh", "eol_cycle"]]
    sources = {"measured_soh": cycles.assign(soh_pred=cycles["soh"])}
    for path in sorted(RESULTS_DIR.glob("predictions_*.csv")):
        pred = pd.read_csv(path)[["cell", "cycle", "soh_pred"]]
        sources[path.stem.removeprefix("predictions_")] = cycles.merge(pred, on=["cell", "cycle"])

    eol_cells = sorted(cycles.loc[cycles.eol_cycle.notna(), "cell"].unique())
    rows, metrics = [], []
    for source, df in sources.items():
        by_window = {w: _rul_for_window(df, w) for w in RUL_WINDOWS}
        for cell in sorted(df["cell"].unique()):
            others = [c for c in eol_cells if c != cell]
            window = min(RUL_WINDOWS, key=lambda w: _rul_error(by_window[w], others)[0])
            chosen = by_window[window][by_window[window].cell == cell]
            rows.append(chosen.assign(source=source, window=str(window or "all")))
            if cell in eol_cells:
                rmse, mae, n = _rul_error(chosen, [cell])
                metrics.append({"source": source, "test_cell": cell, "window": window or "all",
                                "n_cycles": n, "rmse_cycles": rmse, "mae_cycles": mae})

    rul = pd.concat(rows, ignore_index=True)
    rul.to_csv(RESULTS_DIR / "rul_predictions.csv", index=False)
    plot_rul(rul, HEADLINE, FIG_DIR / f"rul_{HEADLINE}.png")

    metrics = pd.DataFrame(metrics)
    means = metrics.groupby("source")[["rmse_cycles", "mae_cycles"]].mean().reset_index()
    metrics = pd.concat([metrics, means.assign(test_cell="mean")], ignore_index=True)
    metrics.to_csv(RESULTS_DIR / "rul_metrics.csv", index=False)
    return metrics


def _print_table(results: pd.DataFrame, metrics, columns=("model", "feature_set")) -> None:
    """Print one pivot table (test cell x columns) per metric."""
    for metric in metrics:
        print(f"{metric}:")
        print(results.pivot_table(index="test_cell", columns=list(columns), values=metric,
                                  sort=False).round(3).to_string())
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one stage of the SOH/RUL pipeline.")
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["eda", "baseline", "lstm", "rul", "all"])
    stage = parser.parse_args().stage

    if stage in ("eda", "all"):
        for path in (plot_soh_curves(build_cycle_table()), plot_discharge_curves()):
            print(f"Saved {path.relative_to(ROOT)}")
        print()
    if stage in ("baseline", "all"):
        print("Random forest and ridge. SOH error in percentage points; the test cell is never")
        print("used for training or model selection.")
        print()
        _print_table(run_baseline(), ("val_rmse_pp", "rmse_pp", "mae_pp"))
    if stage in ("lstm", "all"):
        print("LSTM. SOH error in percentage points (mean of 3 validation-rotation models).")
        print()
        _print_table(run_lstm(), ("val_rmse_pp", "rmse_pp", "mae_pp"))
    if stage in ("rul", "all"):
        print("RUL error in cycles, scored on cycles 10..EOL-1 of cells that reach EOL.")
        print()
        _print_table(run_rul(), ("rmse_cycles", "mae_cycles"), columns=("source",))


if __name__ == "__main__":
    main()
