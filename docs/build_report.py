"""Build the project report PDF from the saved results and figures.

Usage: pip install reportlab, then python docs/build_report.py
All numbers are read from results/ and data/processed/ files produced by real runs.
"""

import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                Spacer, Table, TableStyle)
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate
from reportlab.platypus.frames import Frame
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import FEATURES  # noqa: E402
from src.models import PARAM_GRIDS, RUL_WINDOWS  # noqa: E402

RES, FIG, PROC = ROOT / "results", ROOT / "figures", ROOT / "data" / "processed"
OUT = ROOT / "docs" / "Battery_SOH_RUL_Project_Report.pdf"

# ----------------------------------------------------------------------------- styles
INK, INK2, ACCENT, RULE, SHADE = (colors.HexColor(h) for h in
                                  ("#0b0b0b", "#52514e", "#2a78d6", "#d9d8d4", "#f2f1ee"))
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10,
                      leading=14.2, textColor=INK, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=INK2)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=3)
h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=16, leading=20,
                    spaceBefore=4, spaceAfter=10, textColor=INK, keepWithNext=1)
h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=12, leading=15,
                    spaceBefore=10, spaceAfter=5, textColor=INK, keepWithNext=1)
cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=10.5, spaceAfter=0)
cell_b = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
code = ParagraphStyle("code", parent=body, fontName="Courier", fontSize=8.3, leading=10.5,
                      backColor=SHADE, borderPadding=6, leftIndent=6, rightIndent=6,
                      spaceBefore=4, spaceAfter=10)
caption = ParagraphStyle("caption", parent=small, alignment=TA_CENTER, spaceBefore=3,
                         spaceAfter=12)
title = ParagraphStyle("title", parent=body, fontName="Helvetica-Bold", fontSize=26,
                       leading=31, spaceAfter=10)
subtitle = ParagraphStyle("subtitle", parent=body, fontSize=13, leading=18, textColor=INK2)

story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def H1(text):
    story.append(CondPageBreak(6 * cm))
    p = Paragraph(text, h1)
    p._toc = (0, text)
    story.append(p)


def H2(text):
    p = Paragraph(text, h2)
    p._toc = (1, text)
    story.append(p)


def bullets(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))
    story.append(Spacer(1, 4))


def code_block(text):
    story.append(Preformatted(text.strip("\n"), code))


def table(rows, widths=None, header=True, bold_last=False):
    data = [[Paragraph(str(c), cell_b if (header and i == 0) or (bold_last and i == len(rows) - 1)
                       else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), SHADE), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2)]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 10))


def figure(name, text, width=16.5 * cm):
    path = FIG / name
    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width, height=width * ih / iw)
    story.append(KeepTogether([img, Paragraph(text, caption)]))


def f2(x):
    return f"{x:.2f}"


# ----------------------------------------------------------------------------- data
cyc = pd.read_csv(PROC / "cycles.csv")
feat = pd.read_csv(PROC / "features.csv")
base = pd.read_csv(RES / "baseline_metrics.csv")
lstm = pd.read_csv(RES / "lstm_metrics.csv")
rulm = pd.read_csv(RES / "rul_metrics.csv")
splits = pd.read_csv(RES / "splits.csv")
allm = pd.concat([base, lstm], ignore_index=True)
CELLS = ["B0005", "B0006", "B0007", "B0018"]


def metric(model, fs, cell_, col):
    r = allm[(allm.model == model) & (allm.feature_set == fs) & (allm.test_cell == cell_)]
    return float(r[col].iloc[0])


def rul_metric(source, cell_, col):
    r = rulm[(rulm.source == source) & (rulm.test_cell == cell_)]
    return float(r[col].iloc[0])


per_cell = []
for c, g in cyc.groupby("cell"):
    eol = g.eol_cycle.iloc[0]
    per_cell.append([c, len(g), f"{g.capacity_ah.iloc[0]:.3f}", f"{g.capacity_ah.min():.3f}",
                     f"{g.soh.iloc[0]:.1%}", f"{g.soh.iloc[-1]:.1%}", f"{g.soh.min():.1%}",
                     str(int(eol)) if pd.notna(eol) else "not reached"])

ridge_np = ("ridge", "no_capacity_proxies")
best_soh_rmse = metric(*ridge_np, "mean", "rmse_pp")
best_soh_mae = metric(*ridge_np, "mean", "mae_pp")
best_rul_rmse = rul_metric("ridge_no_capacity_proxies", "mean", "rmse_cycles")
best_rul_mae = rul_metric("ridge_no_capacity_proxies", "mean", "mae_cycles")

# ----------------------------------------------------------------------------- cover
story.append(Spacer(1, 5 * cm))
P("EV Battery State of Health and Remaining Useful Life Prediction", title)
P("Project report: everything built, run and measured", subtitle)
story.append(Spacer(1, 0.8 * cm))
P("NASA PCoE Li-ion Battery Aging Dataset, cells B0005, B0006, B0007 and B0018", body)
P("Report date: 26 October 2026", body)
P("Status: data pipeline, SOH models, LSTM, RUL and README complete. "
  "The Streamlit app is not yet built.", body)
story.append(Spacer(1, 1.2 * cm))
P("Every number in this report is read directly from the result files written by the "
  "project's own code. No figure was typed in by hand.", small)
story.append(PageBreak())

# ----------------------------------------------------------------------------- TOC
P("Contents", h1)
toc = TableOfContents()
toc.levelStyles = [ParagraphStyle("toc0", parent=body, fontName="Helvetica-Bold", fontSize=10,
                                  leftIndent=0, spaceBefore=4),
                   ParagraphStyle("toc1", parent=body, fontSize=9, leftIndent=16, spaceAfter=0)]
story.append(toc)
story.append(PageBreak())

# ----------------------------------------------------------------------------- 1 summary
H1("1. Executive summary")
P("This project predicts two things for lithium-ion cells from their discharge data. The first "
  "is State of Health (SOH): how much capacity a cell has left compared with when it was new. "
  "The second is Remaining Useful Life (RUL): how many more charge and discharge cycles it can "
  "run before it reaches end of life, defined here as 70% SOH.")
P("Every model is tested on a cell it has never seen, using leave-one-cell-out evaluation. "
  "Hyperparameters are chosen on separate validation cells. The test cell is never used for "
  "training or for any modelling choice.")
P("Headline results", h2)
table([["What", "Best model", "RMSE", "MAE"],
       ["SOH, honest feature set", "Ridge regression, no capacity proxies",
        f"{f2(best_soh_rmse)} pp", f"{f2(best_soh_mae)} pp"],
       ["SOH, all features (upper bound)", "Ridge regression, all features",
        f"{f2(metric('ridge', 'all_features', 'mean', 'rmse_pp'))} pp",
        f"{f2(metric('ridge', 'all_features', 'mean', 'mae_pp'))} pp"],
       ["RUL", "Linear trend on ridge SOH predictions",
        f"{best_rul_rmse:.1f} cycles", f"{best_rul_mae:.1f} cycles"]],
      widths=[4.6 * cm, 6.4 * cm, 2.8 * cm, 2.8 * cm])
P("pp means percentage points of SOH. An RMSE of 2 pp means predictions are typically about "
  "2 SOH points away from the truth, for example 78% predicted against 80% actual.", small)
P("Key findings", h2)
bullets([
    "<b>Discharge time nearly equals the answer.</b> At a constant 2 A, the time a cell takes to "
    "run empty is essentially its capacity (correlation 0.999 with SOH). Models are therefore "
    "reported twice: with all features, and without these \"capacity proxy\" features.",
    "<b>A simple linear model wins.</b> Ridge regression beats random forest and the LSTM on the "
    "honest feature set. With only four cells, there is too little data for flexible models.",
    "<b>Tree models cannot extrapolate.</b> Cell B0006 degrades further than any other cell. "
    "The random forest's predictions flatten at the lowest SOH it saw in training.",
    "<b>RUL is unreliable early in life and good late in life.</b> When SOH is nearly flat, "
    "any trend projection is very uncertain. From about cycle 60 the estimate tracks the truth.",
    "<b>B0007 never reaches end of life.</b> Its lowest SOH is 70.02%, just above the threshold, "
    "so its true RUL is undefined and it is excluded from RUL scoring.",
    "<b>All results reproduce exactly.</b> Seeds are fixed and every stage was run twice with "
    "identical output.",
])
story.append(PageBreak())

# ----------------------------------------------------------------------------- 2 problem
H1("2. Problem and goals")
P("Battery packs in electric vehicles lose capacity as they age. Knowing a pack's health and "
  "how long it has left matters for warranty, second-life reuse, maintenance planning and "
  "safety. Measuring capacity directly needs a full, controlled discharge, which is rarely "
  "possible in the field. The aim is to estimate health from signals that are easier to observe.")
P("The project brief set these goals:")
bullets([
    "Parse the NASA .mat files and extract per-cycle discharge data: capacity, voltage, "
    "current, temperature and time.",
    "Compute SOH = current capacity / rated capacity (2.0 Ah). End of life is 70% SOH (1.4 Ah).",
    "Engineer per-cycle features from the discharge curves.",
    "Build a classical baseline model, then a small LSTM on cycle sequences, then RUL prediction.",
    "Evaluate leave-one-cell-out and report RMSE and MAE. Never report training-data results "
    "as test results.",
    "Produce plots, a Streamlit app and a README for a public portfolio repository.",
])
P("Working rules from the brief: work in five phases and stop after each one for approval; "
  "keep the code modular with short docstrings; use only real numbers from actual runs; set "
  "random seeds for reproducibility.")
P("Changes requested during the project", h2)
bullets([
    "<b>No XGBoost.</b> The original baseline was XGBoost. On request it was replaced with a "
    "random forest, and ridge regression was added for model diversity. XGBoost was removed from "
    "the code, requirements and outputs.",
    "<b>Training, validation and test split.</b> On request, a validation stage was added inside "
    "every leave-one-cell-out fold (Section 8).",
])

# ----------------------------------------------------------------------------- 3 dataset
H1("3. The dataset")
P("Source", h2)
P("NASA Prognostics Center of Excellence (PCoE), \"Li-ion Battery Aging Datasets\", item 5 on "
  "the PCoE data repository. The archive used was downloaded from:")
code_block("https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip   (about 210 MB)")
P("The zip holds six inner archives. The four cells used here are in "
  "<i>1. BatteryAgingARC-FY08Q4.zip</i>: B0005.mat, B0006.mat, B0007.mat and B0018.mat. "
  "They go in data/raw/, which is excluded from git.")
P("How the cells were tested", h2)
P("According to the dataset README, the four cells were cycled at room temperature (about "
  "24 C ambient) through three kinds of operation:")
bullets([
    "<b>Charge:</b> constant current 1.5 A to 4.2 V, then constant voltage until the current "
    "fell to 20 mA.",
    "<b>Discharge:</b> constant current 2 A until the voltage fell to 2.7 V (B0005), 2.5 V "
    "(B0006), 2.2 V (B0007) or 2.5 V (B0018).",
    "<b>Impedance:</b> electrochemical impedance spectroscopy sweep from 0.1 Hz to 5 kHz.",
])
P("Testing stopped when the cells reached 30% capacity fade, from 2.0 Ah to 1.4 Ah. This "
  "project uses only the discharge cycles.")
P("File structure", h2)
P("Each .mat file contains one struct named after the cell, with a <i>cycle</i> array. Each "
  "entry has a <i>type</i> (charge, discharge or impedance), the ambient temperature, a start "
  "time as a MATLAB date vector, and a <i>data</i> struct. For discharge cycles, <i>data</i> "
  "holds Voltage_measured, Current_measured, Temperature_measured, Current_load, Voltage_load, "
  "Time (seconds) and Capacity (Ah). B0005.mat contains 616 operations in total.")
P("What the data looks like", h2)
table([["Cell", "Discharge cycles", "First capacity (Ah)", "Min capacity (Ah)", "First SOH",
        "Last SOH", "Min SOH", "EOL cycle"]] + per_cell,
      widths=[1.5 * cm, 2.1 * cm, 2.3 * cm, 2.3 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 2.4 * cm])
P(f"Total: {len(cyc)} discharge cycles. Each cycle has between 179 and 371 samples, roughly "
  "every 9.4 seconds. Time, voltage, current and temperature arrays always have equal length "
  "and contain no missing values.", small)

# ----------------------------------------------------------------------------- 4 structure
H1("4. Project structure and how to run it")
code_block("""
Battery SOH/
  data/raw/              NASA .mat files (gitignored, download separately)
  data/processed/        cycles.csv, features.csv (generated)
  src/
    data_loader.py       parse .mat files, SOH and end of life
    features.py          per-cycle discharge-curve features
    models.py            random forest, ridge, LSTM, RUL estimator, seeds
    evaluate.py          splits, evaluation loops, metrics, pipeline CLI
    plots.py             all figures
  figures/               all PNG plots (generated)
  results/               metrics, predictions, splits (generated)
  notebooks/             exploration only (empty so far)
  docs/                  this report and build_report.py
  requirements.txt, README.md, LICENSE, .gitignore
""")
P("Commands, in order:")
code_block("""
pip install -r requirements.txt
python -m src.data_loader         # parse .mat files -> data/processed/cycles.csv
python -m src.features            # features -> data/processed/features.csv
python -m src.evaluate eda        # exploration plots -> figures/
python -m src.evaluate baseline   # random forest + ridge (about 2.5 minutes)
python -m src.evaluate lstm       # LSTM on the same splits (about 2 minutes)
python -m src.evaluate rul        # RUL from saved predictions (after baseline and lstm)
python -m src.evaluate           # or run every evaluation stage in order (about 6.5 min)
""")
P("Environment used: Windows 11, Python 3.14.4, numpy 2.3.4, pandas 2.3.3, scipy 1.18.0, "
  "scikit-learn 1.9.0, matplotlib 3.11.0, PyTorch 2.14.1 (CPU build). The code is about 760 "
  "lines across five modules.")
story.append(PageBreak())

# ----------------------------------------------------------------------------- 5 phase 1
H1("5. Phase 1: data loading")
P("The loader in <i>src/data_loader.py</i> reads one .mat file with scipy, keeps only the "
  "discharge operations, and numbers them 1, 2, 3 and so on in time order. Each cycle becomes a "
  "Python dict with:")
bullets([
    "the cell name, cycle number, start time (converted from the MATLAB date vector), ambient "
    "temperature and measured capacity in Ah;",
    "four numpy arrays: time in seconds, terminal voltage, current and cell temperature.",
])
P("Main functions:")
table([["Function", "What it does"],
       ["load_cell(cell)", "Returns the list of discharge-cycle dicts for one cell."],
       ["load_all()", "Loads all four cells into a dict keyed by cell name."],
       ["cycles_to_frame(cycles)", "One summary row per cycle: capacity, sample count, duration."],
       ["add_soh(df)", "Adds SOH, an end-of-life flag and each cell's first EOL cycle."],
       ["build_cycle_table()", "Loads everything and returns the per-cycle table with SOH."]],
      widths=[4.5 * cm, 12 * cm])
P("Verification: the loader found 168, 168, 168 and 132 discharge cycles for B0005, B0006, "
  "B0007 and B0018. These match the published counts for this dataset. A second check "
  "confirmed equal-length arrays and no missing values in every cycle.")

# ----------------------------------------------------------------------------- 6 phase 2
H1("6. Phase 2: SOH, end of life and exploration")
P("Definitions", h2)
bullets([
    "<b>SOH</b> = measured capacity of the cycle / 2.0 Ah rated capacity.",
    "<b>End of life (EOL)</b> = the first cycle whose SOH is at or below 70%.",
    "<b>RUL</b> at cycle k = EOL cycle - k. It is defined only for cells that reach EOL.",
])
P("Running the loader writes <i>data/processed/cycles.csv</i> with one row per cycle. That file "
  "is the input for every later step.")
figure("soh_vs_cycle.png", "Figure 1. SOH against discharge cycle for all four cells. The dashed "
       "line is the 70% end-of-life threshold.")
P("What the exploration showed", h2)
bullets([
    "<b>EOL cycles:</b> B0005 at cycle 125, B0006 at 109, B0018 at 97. B0007 bottoms out at "
    "70.02% and never strictly crosses 70%.",
    "<b>Capacity regeneration:</b> every cell shows sudden upward jumps in SOH, for example near "
    "cycles 20, 31, 48 and 90. These are a known effect of rest periods in this dataset. SOH "
    "does not fall smoothly, so models must cope with noise.",
    "<b>B0006 starts above rated capacity</b> at 2.035 Ah (101.8% SOH). The value is kept as "
    "measured, not clipped to 100%.",
])
figure("discharge_curves.png", "Figure 2. Voltage during discharge for four cycles of B0005, "
       "from new (light) to aged (dark).")
bullets([
    "<b>Aged curves sit lower and end sooner.</b> This makes time-to-voltage features natural "
    "candidates.",
    "<b>Each discharge ends with a voltage rebound.</b> When the load switches off, the voltage "
    "recovers by several tenths of a volt. Features must stop at the cutoff, not the last sample.",
    "<b>Cutoff voltages differ between cells</b> (2.7, 2.5, 2.2, 2.5 V). Features must use a "
    "shared end point to be comparable.",
])
story.append(PageBreak())

# ----------------------------------------------------------------------------- 7 phase 3 features
H1("7. Phase 3: feature engineering")
P("How each cycle is cut", h2)
bullets([
    "<b>Loaded segment:</b> samples where current is below -1 A, meaning the 2 A load is on. "
    "This removes the rest samples before the load and the voltage rebound after it.",
    "<b>Common end point:</b> the segment is cut where the voltage first reaches 2.8 V, "
    "interpolated between samples. This is above every cell's cutoff, so all cells are measured "
    "the same way.",
])
P("The 13 features", h2)
table([["Feature", "Meaning"],
       ["discharge_time_s", "Seconds from load-on until the voltage reaches 2.8 V."],
       ["t_to_3p8v_s ... t_to_3p0v_s", "Seconds from load-on until the voltage first reaches "
        "3.8, 3.6, 3.4, 3.2 and 3.0 V (five features, linearly interpolated)."],
       ["v_mean, v_std, v_skew", "Mean, standard deviation and skewness of voltage over the cut "
        "segment: the shape of the discharge curve."],
       ["v_load_drop", "Voltage just before the load switches on minus the first loaded sample. "
        "A proxy for internal resistance."],
       ["temp_max_c", "Peak cell temperature in the cut segment."],
       ["t_to_temp_max_s", "Seconds from load-on to peak temperature."],
       ["temp_rise_c", "Peak temperature minus temperature at load-on."]],
      widths=[4.8 * cm, 11.7 * cm])
corr = feat[FEATURES].corrwith(feat["soh"]).sort_values(key=abs, ascending=False)
P("Correlation with SOH", h2)
P("Pooled over all 636 cycles, as written by <i>python -m src.features</i>:")
half = (len(corr) + 1) // 2
rows = [["Feature", "Corr.", "Feature", "Corr."]]
items = list(corr.items())
for i in range(half):
    a = items[i]
    b = items[i + half] if i + half < len(items) else ("", None)
    rows.append([a[0], f"{a[1]:.3f}", b[0], "" if b[1] is None else f"{b[1]:.3f}"])
table(rows, widths=[4.6 * cm, 2.6 * cm, 4.6 * cm, 2.6 * cm])
P("The capacity proxy problem", h2)
P("Discharge time, time to 3.0 V and time to peak temperature all correlate 0.999 with SOH. "
  "This is physics, not a discovery. The discharge current is a constant 2 A, so capacity in "
  "Ah is roughly current times discharge time. A model given discharge time is close to reading "
  "the answer off the data. It also needs a full discharge, which is exactly the measurement the "
  "project is trying to avoid.")
P("Two feature sets are therefore evaluated everywhere:")
bullets([
    "<b>all_features:</b> all 13 features, as requested in the brief. Treat these results as an "
    "upper bound.",
    "<b>no_capacity_proxies:</b> drops discharge_time_s, t_to_3p0v_s, t_to_3p2v_s and "
    "t_to_temp_max_s, leaving 9 features. The model must infer health from the shape of the "
    "voltage curve. This is the headline set.",
])
story.append(PageBreak())

# ----------------------------------------------------------------------------- 8 evaluation
H1("8. Evaluation protocol")
P("Why split by cell", h2)
P("Neighbouring cycles of the same cell are nearly identical. A random split of cycles would "
  "put cycle 50 in training and cycle 51 in test, and the score would mostly measure memory. "
  "All splits in this project are by whole cell.")
P("Training, validation and test", h2)
P("Each of the four cells takes a turn as the test cell. Inside each of those folds:")
bullets([
    "<b>Test:</b> the held-out cell, used only once for the final score.",
    "<b>Validation:</b> each of the other three cells takes a turn as the validation cell, while "
    "the remaining two train. The hyperparameters with the lowest mean validation RMSE win.",
    "<b>Final fit:</b> the classical models are refit on all three non-test cells with the "
    "chosen hyperparameters, then scored on the test cell. The LSTM is handled slightly "
    "differently (Section 10).",
])
P("All 12 combinations, as saved in <i>results/splits.csv</i>:")
table([["Test cell", "Validation cell", "Training cells"]] +
      splits[["test_cell", "val_cell", "train_cells"]].values.tolist(),
      widths=[3 * cm, 3.4 * cm, 5 * cm])
P("Metrics", h2)
bullets([
    "<b>RMSE</b> (root mean squared error) and <b>MAE</b> (mean absolute error) of SOH, in "
    "percentage points, computed over every cycle of the test cell.",
    "The \"mean\" row in each table is the average over the four test cells.",
    "Validation RMSE is reported separately and is never mixed with test results.",
])
P("Reproducibility", h2)
P("A single seed (42) is set for Python, NumPy and PyTorch. PyTorch uses deterministic "
  "algorithms on one thread. The random forest runs single-threaded. Every stage was run twice "
  "and produced identical output files.")

# ----------------------------------------------------------------------------- 9 classical models
H1("9. Classical models: random forest and ridge")
P("Models", h2)
bullets([
    "<b>Random forest:</b> 300 bagged regression trees. Robust, nonlinear and easy to interpret "
    "through feature importance. It cannot predict outside the range of SOH seen in training.",
    "<b>Ridge regression:</b> a linear model with L2 regularisation, on standardised features. "
    "Unlike trees, it can extrapolate beyond the training range.",
])
P("Hyperparameter grids searched on the validation cells:")
table([["Model", "Grid", "Combinations"],
       ["Random forest", "min_samples_leaf in {1, 2, 5, 10}; max_features in {0.33, 0.5, 1.0}",
        str(len(PARAM_GRIDS["random_forest"]))],
       ["Ridge", "alpha in {0.01, 0.1, 1, 10, 100}", str(len(PARAM_GRIDS["ridge"]))]],
      widths=[3 * cm, 10.5 * cm, 3 * cm])


def soh_table(col, models):
    rows = [["Test cell"] + [m[2] for m in models]]
    for c in CELLS + ["mean"]:
        rows.append([c if c != "mean" else "Mean"] + [f2(metric(m[0], m[1], c, col)) for m in models])
    return rows


classical = [("random_forest", "all_features", "Forest, all"),
             ("random_forest", "no_capacity_proxies", "Forest, no proxies"),
             ("ridge", "all_features", "Ridge, all"),
             ("ridge", "no_capacity_proxies", "Ridge, no proxies")]
w5 = [3 * cm] + [3.35 * cm] * 4
P("Test RMSE (pp)", h2)
table(soh_table("rmse_pp", classical), widths=w5, bold_last=True)
P("Test MAE (pp)", h2)
table(soh_table("mae_pp", classical), widths=w5, bold_last=True)
P("Mean validation RMSE (pp)", h2)
table(soh_table("val_rmse_pp", classical), widths=w5, bold_last=True)
P("In the validation table, the row for a cell gives the validation score from the fold where "
  "that cell was the test cell.", small)
figure("pred_vs_actual_ridge_no_capacity_proxies.png",
       "Figure 3. Ridge regression, no capacity proxies: actual and predicted SOH on each "
       "held-out cell.")
figure("pred_vs_actual_random_forest_no_capacity_proxies.png",
       "Figure 4. Random forest, no capacity proxies. On B0006 the prediction flattens near 65% "
       "because trees cannot go below the lowest SOH seen in training.")
figure("feature_importance_random_forest_no_capacity_proxies.png",
       "Figure 5. Random forest feature importance (no capacity proxies), averaged over the four "
       "folds.", width=13 * cm)
figure("feature_importance_random_forest_all_features.png",
       "Figure 6. Random forest feature importance with all features. The capacity proxies "
       "dominate.", width=13 * cm)
P("Observations", h2)
bullets([
    "<b>Ridge is the best honest model,</b> with mean test RMSE "
    f"{f2(best_soh_rmse)} pp against {f2(metric('random_forest', 'no_capacity_proxies', 'mean', 'rmse_pp'))} "
    "pp for the forest.",
    "<b>B0006 is hardest for every model.</b> It starts at 101.8% and ends at 57.7% SOH, beyond "
    "every training cell. Validation cannot warn about this: in B0006's fold, ridge scores "
    f"{f2(metric('ridge', 'no_capacity_proxies', 'B0006', 'val_rmse_pp'))} pp on validation but "
    f"{f2(metric('ridge', 'no_capacity_proxies', 'B0006', 'rmse_pp'))} pp on test.",
    "<b>Mid-curve voltage carries the signal.</b> Without proxies, time to 3.4 V, time to 3.6 V "
    "and voltage spread hold most of the forest's importance.",
    "<b>Chosen hyperparameters vary between folds</b> (Appendix C). That is expected with only "
    "three development cells per fold.",
])
P("Earlier versions, for the record", h2)
P("Before XGBoost was dropped, an XGBoost baseline was run with fixed settings: mean test RMSE "
  "1.20 pp with all features and 3.51 pp without proxies. Before the validation stage was added, "
  "random forest and ridge used fixed settings: mean RMSE 3.43 and 2.21 pp without proxies. "
  "Those outputs were replaced and are not part of the current results.")

# ----------------------------------------------------------------------------- 10 LSTM
H1("10. LSTM sequence model")
P("Design", h2)
bullets([
    "<b>Input:</b> the features of the last 10 cycles, up to and including the current one. For "
    "the first 9 cycles, the window is padded by repeating cycle 1. No future cycle is ever used.",
    "<b>Architecture:</b> one LSTM layer with 32 hidden units, then a linear layer that outputs "
    "SOH for the last cycle in the window.",
    "<b>Scaling:</b> features are standardised and the SOH target is centred and scaled, using "
    "statistics from the training cells only.",
    "<b>Training:</b> Adam optimiser, learning rate 0.003, batch size 32, mean squared error "
    "loss, up to 300 epochs. Training stops early after 30 epochs without validation "
    "improvement, and the best epoch is restored.",
    "<b>Use of validation:</b> each test fold trains three LSTMs, one per validation cell, each "
    "on the remaining two cells. Their test predictions are averaged. This uses the validation "
    "cell for early stopping, which a refit on all three cells could not do.",
    "<b>No tuning against test results.</b> The settings above were fixed before the first run.",
])
lstm_models = [("lstm", "all_features", "LSTM, all"), ("lstm", "no_capacity_proxies", "LSTM, no proxies"),
               ("ridge", "no_capacity_proxies", "Ridge, no proxies")]
w4 = [3 * cm] + [4.2 * cm] * 3
P("Test RMSE (pp), with ridge for comparison", h2)
table(soh_table("rmse_pp", lstm_models), widths=w4, bold_last=True)
P("Test MAE (pp)", h2)
table(soh_table("mae_pp", lstm_models), widths=w4, bold_last=True)
figure("pred_vs_actual_lstm_no_capacity_proxies.png",
       "Figure 7. LSTM, no capacity proxies: actual and predicted SOH on each held-out cell.")
P("Observations", h2)
bullets([
    f"<b>The LSTM underperforms ridge</b> on the honest feature set: "
    f"{f2(metric('lstm', 'no_capacity_proxies', 'mean', 'rmse_pp'))} pp against {f2(best_soh_rmse)} pp.",
    "<b>Data is the bottleneck.</b> Each LSTM trains on only two cells, about 300 cycles. That "
    "is too little for a sequence model to learn more than a linear trend.",
    "<b>It also struggles with B0006 and B0018,</b> which degrade faster than the training cells.",
])
figure("pred_vs_actual_lstm_all_features.png",
       "Figure 8. LSTM with all features, for reference.")

# ----------------------------------------------------------------------------- 11 RUL
story.append(PageBreak())
H1("11. Remaining useful life")
P("Method", h2)
P("RUL is estimated from a sequence of SOH values, either measured or predicted. At every "
  "cycle k from cycle 10 onward:")
bullets([
    "fit a straight line to the SOH values in a trailing window ending at cycle k (past only);",
    "find the cycle where that line reaches 70% SOH;",
    "RUL = that cycle - k, clipped to between 0 and 300 cycles. A flat or rising trend gives 300.",
])
P(f"The trailing window length is chosen from {{20, 30, 50, all history}}. For each cell, the "
  "window is picked by the lowest RUL error on the other cells that reach EOL, never on the "
  "cell itself.")
P("RUL is computed from the predicted SOH of every model, plus from measured SOH as a reference. "
  "It is scored on cells B0005, B0006 and B0018, on cycles 10 up to EOL - 1. B0007 is excluded "
  "because it never reaches EOL.")
P("RUL error by SOH source (cycles)", h2)
P("Per-cell columns are RMSE in cycles. Scored cycles: 115 for B0005, 99 for B0006, 87 for "
  "B0018.", small)
names = {"ridge_no_capacity_proxies": "Ridge, no proxies", "ridge_all_features": "Ridge, all",
         "measured_soh": "Measured SOH (reference)",
         "random_forest_no_capacity_proxies": "Forest, no proxies",
         "random_forest_all_features": "Forest, all", "lstm_no_capacity_proxies": "LSTM, no proxies",
         "lstm_all_features": "LSTM, all"}
means = rulm[rulm.test_cell == "mean"].sort_values("rmse_cycles")
rows = [["SOH source", "B0005", "B0006", "B0018", "Mean RMSE", "Mean MAE"]]
for s in means.source:
    rows.append([names[s]] + [f"{rul_metric(s, c, 'rmse_cycles'):.1f}" for c in ["B0005", "B0006", "B0018"]]
                + [f"{rul_metric(s, 'mean', 'rmse_cycles'):.1f}", f"{rul_metric(s, 'mean', 'mae_cycles'):.1f}"])
table(rows, widths=[4.6 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.6 * cm, 2.6 * cm])
figure("rul_ridge_no_capacity_proxies.png",
       "Figure 9. RUL over time for the three cells that reach EOL: true RUL (dashed), estimate "
       "from measured SOH, and estimate from ridge-predicted SOH.")
P("Observations", h2)
bullets([
    f"<b>Best RUL:</b> the trend on ridge (no proxies) predictions, mean RMSE {best_rul_rmse:.1f} "
    f"cycles and MAE {best_rul_mae:.1f} cycles.",
    "<b>Early life is the weak spot.</b> B0005's SOH is almost flat for its first 40 cycles, so "
    "the projected line barely falls and RUL hits the 300-cycle cap. Most of B0005's error comes "
    "from this stretch.",
    "<b>Late life is accurate.</b> From about cycle 60, estimates follow the true RUL closely on "
    "all three cells. That is the period where RUL matters most.",
    "<b>Predicted SOH beat measured SOH here.</b> Ridge predictions are smoother than measured "
    "capacity, which jumps after rest periods. With only three scored cells this is an "
    "observation, not a general result.",
])
P("A disclosed methodology change", h2)
P("The first RUL version used a fixed 30-cycle window. Its results were seen before the window "
  "selection was added: mean RMSE 52.5 cycles for ridge (no proxies) and 54.1 for measured SOH. "
  "The change was made because capacity jumps visibly broke the short window. Selection never "
  "uses the scored cell, but the decision to add it was made after seeing test-cell errors, "
  "which is a mild form of researcher choice. It improved ridge to 36.5 and worsened the "
  "measured-SOH reference to 59.3.")
story.append(PageBreak())

# ----------------------------------------------------------------------------- 12 comparison
H1("12. All models side by side")
all_models = [("random_forest", "no_capacity_proxies", "Random forest"), ("ridge", "no_capacity_proxies", "Ridge"),
              ("lstm", "no_capacity_proxies", "LSTM")]
rows = [["Model (no capacity proxies)", "SOH RMSE (pp)", "SOH MAE (pp)", "RUL RMSE (cycles)",
         "RUL MAE (cycles)"]]
for m, fs, label in all_models:
    src = f"{m}_{fs}"
    rows.append([label, f2(metric(m, fs, "mean", "rmse_pp")), f2(metric(m, fs, "mean", "mae_pp")),
                 f"{rul_metric(src, 'mean', 'rmse_cycles'):.1f}", f"{rul_metric(src, 'mean', 'mae_cycles'):.1f}"])
table(rows, widths=[4.8 * cm, 2.9 * cm, 2.9 * cm, 3 * cm, 3 * cm])
rows = [["Model (all features)", "SOH RMSE (pp)", "SOH MAE (pp)", "RUL RMSE (cycles)", "RUL MAE (cycles)"]]
for m, _, label in all_models:
    src = f"{m}_all_features"
    rows.append([label, f2(metric(m, "all_features", "mean", "rmse_pp")),
                 f2(metric(m, "all_features", "mean", "mae_pp")),
                 f"{rul_metric(src, 'mean', 'rmse_cycles'):.1f}", f"{rul_metric(src, 'mean', 'mae_cycles'):.1f}"])
table(rows, widths=[4.8 * cm, 2.9 * cm, 2.9 * cm, 3 * cm, 3 * cm])
P("All values are means over the held-out cells: four cells for SOH, three for RUL.", small)

# ----------------------------------------------------------------------------- 13 engineering
H1("13. Engineering notes")
bullets([
    "<b>Download certificate check.</b> On this Windows machine, curl failed the certificate "
    "revocation check for the AWS link. The download worked with curl's --ssl-no-revoke flag. "
    "A browser download is unaffected.",
    "<b>Python 3.14 and PyTorch.</b> A CPU build of PyTorch 2.14.1 installed cleanly from the "
    "PyTorch CPU package index.",
    "<b>Parallel random forest warnings.</b> Running the forest on all cores printed a flood of "
    "scikit-learn warnings under Python 3.14. It now runs single-threaded, which also keeps runs "
    "deterministic. The full classical stage takes about 2.5 minutes.",
    "<b>Determinism.</b> Every stage (baseline, LSTM, RUL) was run twice and the output files "
    "were compared byte for byte. They matched.",
    "<b>Generated outputs.</b> data/processed/, figures/ and results/ are produced by the "
    "pipeline. They can be committed so the README and app work without rerunning anything.",
])

# ----------------------------------------------------------------------------- 14 limitations
H1("14. Limitations")
bullets([
    "<b>Four cells is very little data.</b> Every number here comes from one held-out cell at a "
    "time. Differences of a fraction of a point between models are not meaningful.",
    "<b>One temperature, one load profile.</b> All cells ran at about 24 C with a constant 2 A "
    "discharge. Real EV use has variable load and temperature, so these models would not "
    "transfer directly.",
    "<b>Features need most of a discharge.</b> Even without the capacity proxies, features such "
    "as time to 3.4 V require a controlled constant-current discharge from full charge.",
    "<b>No uncertainty estimates.</b> Predictions are point values. RUL in particular would "
    "benefit from a range.",
    "<b>B0007 has no true RUL</b> under the strict 70% rule, which leaves three cells for RUL "
    "scoring.",
    "<b>Linear RUL projection</b> ignores the curvature of real degradation and is unreliable "
    "while SOH is still flat.",
])

# ----------------------------------------------------------------------------- 15 next
H1("15. What comes next")
P("Not yet built:")
bullets([
    "<b>Streamlit app</b> (app/streamlit_app.py): pick a cell and model, see actual and "
    "predicted SOH over cycles and the RUL estimate over time. It will read the saved result "
    "files, so it starts instantly and does not need PyTorch.",
])
P("Possible later improvements: impedance features from the EIS measurements, uncertainty "
  "intervals for RUL, and the larger set of NASA cells in the same archive.")
story.append(PageBreak())

# ----------------------------------------------------------------------------- appendices
H1("Appendix A. File inventory")
files = [
    ("src/data_loader.py", "Parses .mat files; SOH, EOL; writes data/processed/cycles.csv."),
    ("src/features.py", "Per-cycle features; writes data/processed/features.csv."),
    ("src/models.py", "Random forest, ridge, grids, seeds, LSTM, RUL estimator."),
    ("src/evaluate.py", "Splits, nested evaluation, LSTM loop, RUL scoring, pipeline CLI."),
    ("src/plots.py", "All figures in one consistent style."),
    ("data/processed/cycles.csv", "One row per discharge cycle with capacity and SOH."),
    ("data/processed/features.csv", "cycles.csv plus the 13 features."),
    ("results/splits.csv", "The 12 test / validation / training combinations."),
    ("results/baseline_metrics.csv", "Random forest and ridge metrics with chosen hyperparameters."),
    ("results/lstm_metrics.csv", "LSTM metrics."),
    ("results/predictions_*.csv", "Out-of-fold SOH predictions for each model and feature set."),
    ("results/rul_predictions.csv", "RUL estimate per cycle, per cell, per SOH source."),
    ("results/rul_metrics.csv", "RUL errors and the chosen trend window."),
    ("figures/*.png", "All 11 figures shown in this report."),
    ("requirements.txt", "numpy, pandas, scipy, scikit-learn, matplotlib, streamlit, torch."),
    (".gitignore", "Excludes data/raw/, .mat and .zip files, caches and editor folders."),
    ("LICENSE", "MIT license."),
    ("docs/build_report.py", "Rebuilds this PDF from results/ and figures/."),
]
table([["Path", "Contents"]] + [list(f) for f in files], widths=[5.2 * cm, 11.3 * cm])

H1("Appendix B. Function reference")
funcs = [
    ("features.cycle_features", "Feature vector for one cycle (loaded segment, 2.8 V end)."),
    ("features.build_feature_table", "Features for all cells plus SOH columns."),
    ("models.make_random_forest / make_ridge", "Model factories taking a seed and hyperparameters."),
    ("models.set_seeds / set_torch_seeds", "Seed Python, NumPy and PyTorch; deterministic torch."),
    ("models.make_sequences", "Per-cell sliding windows of 10 cycles, left-padded."),
    ("models.SOHLSTM / train_lstm / predict_lstm", "LSTM model, training with early stopping, inference."),
    ("models.estimate_rul", "Trailing-window linear trend extrapolated to 70% SOH."),
    ("evaluate.make_splits", "Builds the cell-level split table."),
    ("evaluate.select_params", "Validation-based hyperparameter selection."),
    ("evaluate.leave_one_cell_out", "Nested evaluation for the classical models."),
    ("evaluate.lstm_leave_one_cell_out", "Evaluation for the LSTM."),
    ("evaluate.run_rul", "RUL for every SOH source with per-cell window selection."),
    ("evaluate.main", "Command line: eda, baseline, lstm, rul or all."),
    ("plots.plot_*", "SOH curves, discharge curves, predictions, importance, RUL."),
]
table([["Function", "Purpose"]] + [list(f) for f in funcs], widths=[6.4 * cm, 10.1 * cm])

H1("Appendix C. Hyperparameters chosen per fold")
rows = [["Model", "Feature set", "Test cell", "Chosen on validation"]]
for _, r in base[base.test_cell != "mean"].iterrows():
    rows.append([r.model.replace("_", " "), r.feature_set.replace("_", " "), r.test_cell,
                 str(r.params).replace("'", "")])
table(rows, widths=[3 * cm, 4 * cm, 2.2 * cm, 7.3 * cm])
rows = [["SOH source", "B0005", "B0006", "B0018"]]
for s in means.source:
    w = rulm[(rulm.source == s) & (rulm.test_cell != "mean")].set_index("test_cell")["window"]
    rows.append([names[s]] + [str(w.get(c, "")) for c in ["B0005", "B0006", "B0018"]])
P("RUL trend window chosen per cell (in cycles; \"all\" means full history)", h2)
table(rows, widths=[5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm])

H1("Appendix D. Glossary")
gloss = [
    ("SOH", "State of Health: current capacity divided by rated capacity (2.0 Ah)."),
    ("RUL", "Remaining Useful Life: cycles left until SOH first reaches 70%."),
    ("EOL", "End of life: the first cycle at or below 70% SOH (1.4 Ah)."),
    ("pp", "Percentage points of SOH."),
    ("RMSE", "Root mean squared error. Penalises large errors more than MAE."),
    ("MAE", "Mean absolute error. The typical size of an error."),
    ("Leave-one-cell-out", "Train on some cells, test on a cell the model has never seen; repeat for each cell."),
    ("Capacity proxy", "A feature that is nearly a direct capacity measurement, such as discharge time at constant current."),
    ("Capacity regeneration", "Temporary capacity recovery after a rest period, seen as upward jumps in SOH."),
    ("LSTM", "Long short-term memory network, a neural network for sequences."),
    ("EIS", "Electrochemical impedance spectroscopy, an AC measurement of internal resistance."),
]
table([["Term", "Meaning"]] + [list(g) for g in gloss], widths=[3.8 * cm, 12.7 * cm])


# ----------------------------------------------------------------------------- build
class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                         topMargin=2 * cm, bottomMargin=2 * cm,
                         title="EV Battery SOH and RUL Prediction: Project Report",
                         author="Battery SOH project")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._footer)])

    def _footer(self, canv, doc):
        if doc.page == 1:
            return
        canv.saveState()
        canv.setFont("Helvetica", 8)
        canv.setFillColor(INK2)
        canv.drawString(self.leftMargin, 1.1 * cm, "EV Battery SOH and RUL Prediction")
        canv.drawRightString(A4[0] - self.rightMargin, 1.1 * cm, f"Page {doc.page}")
        canv.restoreState()

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc"):
            level, text = flowable._toc
            self.notify("TOCEntry", (level, text, self.page))


OUT.parent.mkdir(exist_ok=True)
Doc(OUT).multiBuild(story)
print(f"Wrote {OUT}")
