"""Central configuration: paths, column groups and the experimental constants.

Every script imports from here so that one change (e.g. the horizon N) is applied
consistently to the whole pipeline.
"""
from pathlib import Path
import os
import sys

# print unicode (Δ, ≤, °C) safely on Windows consoles and redirected outputs
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]          # step2_standard_solutions/
REPO = ROOT.parent                                   # repository root
DATA_CSV = Path(os.environ.get("UR3_CSV", REPO / "UR3CobotOps.csv"))

OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
MODELS = OUT / "models"
for _p in (FIG, TAB, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- column groups
CUR = [f"Current_J{i}" for i in range(6)]
TEM = [f"Temperature_J{i}" for i in range(6)]
SPD = [f"Speed_J{i}" for i in range(6)]
TOOL = ["Tool_current"]
SENSORS = CUR + TEM + SPD + TOOL                     # 19 raw sensor channels

# ---------------------------------------------------------------- problem constants
HORIZON = 3          # N: predict a grip-loss onset within the next N seconds
WINDOW = 5           # length (s) of the past window used by rolling features
GAP = HORIZON + WINDOW  # embargo (rows) between train / val / test and between CV folds
SEGMENT_BREAK_S = 5.0   # a time jump larger than this starts a new segment
MERGE_GAP = 2        # two grip-loss events separated by <= MERGE_GAP normal rows are merged
FFILL_LIMIT = 2      # causal forward-fill of empty sensor rows, at most 2 consecutive seconds
SPLIT = (0.70, 0.15, 0.15)   # chronological train / validation / test
N_SPLITS_CV = 5      # TimeSeriesSplit folds inside the training period
N_BOOT = 1000        # bootstrap replicates (resampling whole cycles)
MAX_FA_PER_HOUR = 30.0  # alert budget of the 'trustworthy' operating point (~1 false alarm / 2 min)
