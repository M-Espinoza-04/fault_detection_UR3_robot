"""Collect every CSV table of outputs/tables into one JSON file used by build_report.js."""
import json
from pathlib import Path

import pandas as pd

TAB = Path(__file__).resolve().parents[1] / "outputs" / "tables"
out = {}
for f in sorted(TAB.glob("*.csv")):
    df = pd.read_csv(f)
    df = df.astype(object).where(pd.notna(df), None)
    out[f.stem] = {"columns": list(df.columns), "rows": df.values.tolist()}
    idx = pd.read_csv(f, index_col=0)
    out[f.stem]["index"] = [str(i) for i in idx.index]
Path(__file__).with_name("report_data.json").write_text(json.dumps(out, default=str, indent=1))
print(len(out), "tables")
