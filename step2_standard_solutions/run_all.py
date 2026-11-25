"""Run the whole Step-2 pipeline in order. Usage: python run_all.py [--fast]"""
import subprocess
import sys
import time

steps = ["01_data_analysis.py", "02_learning_plan.py", "03_baseline.py", "04_sota_benchmark.py", "05_automl.py"]
for s in steps:
    t = time.time()
    print(f"\n===== {s} =====", flush=True)
    args = [sys.executable, s] + (["--fast"] if ("--fast" in sys.argv and s.startswith("05")) else [])
    subprocess.run(args, check=True)
    print(f"----- {s} done in {time.time() - t:.0f} s", flush=True)
