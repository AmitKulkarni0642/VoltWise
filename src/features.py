"""Per-cycle features engineered from discharge curves.

Each cycle is cut to the loaded part of the discharge (current below -1 A) and then
to a common 2.8 V end voltage. The cells were run to different cutoffs
(2.7, 2.5, 2.2, 2.5 V), so a shared end point keeps features comparable.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew

from src.data_loader import CELLS, PROCESSED_DIR, add_soh, cycles_to_frame, load_all

LOAD_CURRENT_A = -1.0
END_VOLTAGE_V = 2.8
THRESHOLDS_V = [3.8, 3.6, 3.4, 3.2, 3.0]


def _threshold_name(level: float) -> str:
    """Feature name for the time to reach a voltage, e.g. 3.4 -> 't_to_3p4v_s'."""
    return f"t_to_{str(level).replace('.', 'p')}v_s"


FEATURES = (
    ["discharge_time_s"]
    + [_threshold_name(v) for v in THRESHOLDS_V]
    + ["v_mean", "v_std", "v_skew", "v_load_drop",
       "temp_max_c", "t_to_temp_max_s", "temp_rise_c"]
)


def _time_to_voltage(t: np.ndarray, v: np.ndarray, level: float) -> float:
    """First time the voltage falls to `level`, linearly interpolated (NaN if never)."""
    below = np.flatnonzero(v <= level)
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    t0, t1, v0, v1 = t[i - 1], t[i], v[i - 1], v[i]
    return float(t0 + (v0 - level) * (t1 - t0) / (v0 - v1))


def cycle_features(cycle: dict) -> dict:
    """Compute the feature vector for one discharge cycle."""
    t, v = cycle["time_s"], cycle["voltage_v"]
    i, temp = cycle["current_a"], cycle["temperature_c"]

    loaded = np.flatnonzero(i < LOAD_CURRENT_A)
    start, stop = loaded[0], loaded[-1] + 1
    t, v, temp = t[start:stop] - t[start], v[start:stop], temp[start:stop]

    t_end = _time_to_voltage(t, v, END_VOLTAGE_V)
    window = t <= t_end if np.isfinite(t_end) else np.ones_like(t, dtype=bool)
    v_w, temp_w, t_w = v[window], temp[window], t[window]

    feats = {"discharge_time_s": t_end}
    for level in THRESHOLDS_V:
        feats[_threshold_name(level)] = _time_to_voltage(t, v, level)
    feats.update(
        v_mean=float(v_w.mean()),
        v_std=float(v_w.std()),
        v_skew=float(skew(v_w)),
        # Voltage step when the load switches on, a proxy for internal resistance.
        v_load_drop=float(cycle["voltage_v"][max(start - 1, 0)] - v[0]),
        temp_max_c=float(temp_w.max()),
        t_to_temp_max_s=float(t_w[np.argmax(temp_w)]),
        temp_rise_c=float(temp_w.max() - temp_w[0]),
    )
    return feats


def build_feature_table(cells=CELLS) -> pd.DataFrame:
    """One row per discharge cycle: identifiers, SOH target, and features."""
    frames = []
    for cell, cycles in load_all(cells).items():
        base = cycles_to_frame(cycles)
        feats = pd.DataFrame([cycle_features(c) for c in cycles])
        frames.append(pd.concat([base, feats], axis=1))
    return add_soh(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    table = build_feature_table()
    out = PROCESSED_DIR / "features.csv"
    table.to_csv(out, index=False)
    print(f"Saved {table.shape[0]} rows x {len(FEATURES)} features to {out.name}\n")

    nan_counts = table[FEATURES].isna().sum()
    print("Missing values per feature:", dict(nan_counts[nan_counts > 0]) or "none")
    print("\nCorrelation of each feature with SOH (all cells pooled):")
    corr = table[FEATURES].corrwith(table["soh"]).sort_values(key=abs, ascending=False)
    print(corr.round(3).to_string())
