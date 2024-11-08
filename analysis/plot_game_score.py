import argparse
import pathlib
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

def read_json(path):
    events = []
    with (open(path, "r")) as openfile:
        for line in openfile:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error {e} in line: {line}")
                break
    return events

def compute_success_rates(runs, budget=1e6, sortby=None):
    new_runs = []
    for run in runs:
        # Get the cols that are start with achivement
        achievement_cols = [col for col in run.columns if col.startswith("achievement")]
        new_df = pd.DataFrame(columns=achievement_cols)
        # Get the mean for each achievement if the column is at least 1
        for col in achievement_cols:
            print(f'SUM: {run[col].sum()}')
            if run[col].sum() > 0:
                print(f'MEAN: {run[col].mean()}')
                new_df = pd.concat([new_df[col], 100 * run[col].mean()])
            else:
                new_df[col] = 0
        print(new_df)
        new_runs.append(new_df)
    
    print(new_runs[0].head())
        
    pass

def compute_scores(percents):
  # Geometric mean with an offset of 1%.
  assert (0 <= percents).all() and (percents <= 100).all()
  if (percents <= 1.0).all():
    print('Warning: The input may not be in the right range.')
  with warnings.catch_warnings():  # Empty seeds become NaN.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
  return scores

def read_crafter_json(indir, _filename, clip = True):
    indir = pathlib.Path(indir)
    filenames = sorted(list(indir.glob("**/*/stats.jsonl")))
    runs = []
    for idx, fn in enumerate(filenames):
        df = pd.DataFrame(data=read_json(fn))
        df["run"] = idx
        runs.append(df)

    print(f"Loaded {len(runs)} runs.")
    compute_success_rates(runs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default="logdir/random_agent",
        help="Path to the folder containing different runs.",
    )
    parser.add_argument(
        "--filename",
        default="demo_plot",
        help="Name of the output file.",
    )
    cfg = parser.parse_args()

    read_crafter_json(cfg.logdir, cfg.filename)

