"""Load discharge cycles from the NASA PCoE Li-ion battery aging .mat files."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

CELLS = ["B0005", "B0006", "B0007", "B0018"]
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

RATED_CAPACITY_AH = 2.0
EOL_SOH = 0.70

SIGNALS = {
    "Time": "time_s",
    "Voltage_measured": "voltage_v",
    "Current_measured": "current_a",
    "Temperature_measured": "temperature_c",
}


def _matlab_datevec_to_datetime(vec) -> datetime:
    """Convert a MATLAB date vector [Y, M, D, h, m, s] to a datetime."""
    y, mo, d, h, mi, s = (float(x) for x in vec)
    return datetime(int(y), int(mo), int(d), int(h), int(mi)) + timedelta(seconds=s)


def load_cell(cell: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    """Return the discharge cycles of one cell, in chronological order.

    Each cycle is a dict with: cell, cycle (1-based discharge index),
    start_time, ambient_temp_c, capacity_ah, and numpy arrays
    time_s, voltage_v, current_a, temperature_c.
    """
    path = Path(raw_dir) / f"{cell}.mat"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See README for download steps.")

    mat = sio.loadmat(path, simplify_cells=True)
    raw_cycles = mat[cell]["cycle"]

    cycles = []
    for op in raw_cycles:
        if op["type"] != "discharge":
            continue
        data = op["data"]
        cycle = {
            "cell": cell,
            "cycle": len(cycles) + 1,
            "start_time": _matlab_datevec_to_datetime(op["time"]),
            "ambient_temp_c": float(op["ambient_temperature"]),
            "capacity_ah": float(data["Capacity"]),
        }
        for src, dst in SIGNALS.items():
            cycle[dst] = np.asarray(data[src], dtype=float).ravel()
        cycles.append(cycle)
    return cycles


def load_all(cells=CELLS, raw_dir: Path = RAW_DIR) -> dict[str, list[dict]]:
    """Load discharge cycles for all cells, keyed by cell name."""
    return {cell: load_cell(cell, raw_dir) for cell in cells}


def cycles_to_frame(cycles: list[dict]) -> pd.DataFrame:
    """Summarise cycles as one row per discharge cycle (no raw time series)."""
    rows = [
        {
            "cell": c["cell"],
            "cycle": c["cycle"],
            "start_time": c["start_time"],
            "ambient_temp_c": c["ambient_temp_c"],
            "capacity_ah": c["capacity_ah"],
            "n_samples": len(c["time_s"]),
            "duration_s": c["time_s"][-1] - c["time_s"][0],
        }
        for c in cycles
    ]
    return pd.DataFrame(rows)


def add_soh(df: pd.DataFrame) -> pd.DataFrame:
    """Add SOH, end-of-life flag, and the cell's first EOL cycle (NaN if never reached).

    SOH = capacity / rated capacity. EOL is the first cycle with SOH <= 70%.
    """
    df = df.copy()
    df["soh"] = df["capacity_ah"] / RATED_CAPACITY_AH
    df["below_eol"] = df["soh"] <= EOL_SOH
    eol = df[df["below_eol"]].groupby("cell")["cycle"].min()
    df["eol_cycle"] = df["cell"].map(eol)
    return df


def build_cycle_table(cells=CELLS, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load all cells and return one row per discharge cycle, with SOH columns."""
    data = load_all(cells, raw_dir)
    df = pd.concat([cycles_to_frame(c) for c in data.values()], ignore_index=True)
    return add_soh(df)


if __name__ == "__main__":
    table = build_cycle_table()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "cycles.csv"
    table.to_csv(out, index=False)

    print(f"Saved {len(table)} rows to {out.relative_to(ROOT)}\n")
    print(f"{'cell':6} {'cycles':>6} {'SOH first':>9} {'SOH last':>8} {'SOH min':>7}  EOL cycle")
    for cell, g in table.groupby("cell"):
        eol = g["eol_cycle"].iloc[0]
        eol_txt = f"{int(eol)}" if pd.notna(eol) else "not reached"
        print(f"{cell:6} {len(g):6d} {g.soh.iloc[0]:9.1%} {g.soh.iloc[-1]:8.1%} "
              f"{g.soh.min():7.1%}  {eol_txt}")
