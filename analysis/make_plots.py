import pickle
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np

import warnings

def process_compute_success_rates(filespaths):
    for file_path in filespaths:
        logs = read_crafter_log_json(file_path, clip=False)
        compute_success_rates(logs)

def compare_all_success_rates(filespaths):
    succ_logs = []
    plt.figure(figsize=(10, 10))
    for idx,file_path in enumerate(filespaths):
        logs = read_crafter_log_json(file_path, clip=False)
        succ_logs.append(compute_success_rates(logs))

    # get the name of the runs
    # print(succ_logs)
    name_runs = [log['run_name'].iloc[0] for log in succ_logs]
    print(name_runs)

    # Compute the mean of the success rates for each run
    log = pd.concat(succ_logs, ignore_index=True)
    # For each run get the best success rate
    best_success_rate = log.groupby('run_name').max()
    # Have the achivement as the index
    best_success_rate = best_success_rate.T
    # Crate a dictionary with the best success rate for each run
    # Show only the last plot 
    plt.figure(figsize=(10, 10), layout='constrained')
    best_success_rate.plot.bar(rot=90)
    plt.title('Best success rate for each run')
    plt.ylabel('Success rate')
    plt.savefig('logdir/best_success_rate.png')


def compute_success_rates(df, budget=1e6, sortby=None):
    name = df['run_name'].iloc[0]
    achivement_list = [col for col in df.columns if col.startswith('achievement_')]

    success_rates = pd.DataFrame()
    for achievement in achivement_list:
        success_rates[achievement] = df.groupby(['run_name', 'run'])[achievement].apply(lambda x: (x >= 1).sum()) / df.groupby(['run_name', 'run']).size()

    # Cut the achivement from the name
    success_rates.columns = [col.replace('achievement_', '') for col in success_rates.columns]
    success_rates['run_name'] = name

    # Print a barplot of the success rates
    plt.figure(figsize=(10, 10))
    sns.barplot(data=success_rates, orient='v')
    plt.xticks(rotation=90, ha='right')
    plt.title(f'Success rates for {name}')
    plt.ylabel('Success rate')
    plt.savefig(f'logdir/success_rates_{name}.png')

    return success_rates


def read_pkl(path):
    events = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                events.append(pickle.load(openfile))
            except EOFError:
                break
    return events

def read_json(path):
    events = []
    with (open(path, "r")) as openfile:
        for line in openfile:
            events.append(json.loads(line))
    return events

def read_crafter_log_json(indir, clip=True):
    indir = pathlib.Path(indir)
    name_run = indir.parts[-1]
    print(name_run)
    filenames = sorted(list(indir.glob("**/*/stats.jsonl")))
    runs = []
    for idx, fn in enumerate(filenames):
        df = pd.DataFrame(data=read_json(fn))
        df["run_name"] = name_run
        df["run"] = idx
        runs.append(df)

    if clip:
        min_len = min([len(run) for run in runs])
        runs = [run[:min_len] for run in runs]
        print(f"Clipped all runs to {min_len}.")
    df = pd.concat(runs, ignore_index=True)
    return df

def read_crafter_logs(indir, clip=True):
    indir = pathlib.Path(indir)
    name_run = indir.parts[-1]
    print(name_run)
    # read the pickles
    filenames = sorted(list(indir.glob("**/*/eval_stats.pkl")))
    runs = []
    for idx, fn in enumerate(filenames):
        df = pd.DataFrame(columns=["step", "avg_return", "run_name"], data=read_pkl(fn))
        df["run_name"] = name_run
        df["run"] = idx
        runs.append(df)
    df = pd.concat(runs, ignore_index=True)
    return df

def read_logs(filespaths):
    logs_list = []
    for file_path in filespaths:
        logs = read_crafter_logs(file_path)
        logs_list.append(logs)
    return logs_list

def comparison(filepaths, name_save = "comparison_all_runs.png"):
    logs_list = read_logs(filepaths)
    # Extract from the filepaths the name of the run
    name_runs = [pathlib.Path(file).parts[-1] for file in filepaths]
    custom_palette = {name: sns.color_palette()[idx] for idx, name in enumerate(name_runs)}

    plt.figure(figsize=(8, 6))
    for logs in logs_list:
        sns.lineplot(x="step", y="avg_return", data=logs, hue='run_name', palette=custom_palette)
        # Add label and title
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
    plt.savefig(f'logdir/{name_save}')
    plt.show()

def compare_with_random(filepath, name_log, full=False):
    if full:
        filepaths = ['logdir/random_agent-1M/', filepath]
        custom_palette = {"random_agent-1M": "#1f77b4", name_log: "#ff7f0e"}
    else:
        filepaths = ['logdir/random_agent-100k/', filepath]
        custom_palette = {"random_agent-100k": "#1f77b4", name_log: "#ff7f0e"}

    log_list = read_logs(filepaths)

    plt.figure(figsize=(8, 6))
    for idx, logs in enumerate(log_list):
        sns.lineplot(x="step", y="avg_return", data=logs, hue='run_name', palette=custom_palette)
        # Add label and title
        plt.xlabel('Step')
        plt.ylabel('Average Reward')
        plt.title('Random vs ' + name_log)
    plt.savefig('logdir/random_vs_' + name_log + '.png')

def plot_counts_eps(filepath, name_run):
    log_list = read_crafter_log_json(filepath)
    log_list = log_list[log_list['length'] <= 1e6]

if __name__ == "__main__":
    # filespaths = ['logdir/a2c_new/', 'logdir/ppo_attention_new/', 'logdir/reinforce/']

    compare_with_random('logdir/a2c-100k/', 'a2c-100k')
    compare_with_random('logdir/ppo_attention-100k/', 'ppo_attention-100k')
    compare_with_random('logdir/reinforce-100k/', 'reinforce-100k')

    compare_with_random('logdir/a2c-1M/', 'a2c-1M', full=True)
    compare_with_random('logdir/ppo_attention-1M/', 'ppo_attention-1M', full=True)

    filespaths = ['logdir/a2c-100k/', 'logdir/ppo_attention-100k/', 'logdir/reinforce-100k/', 'logdir/random_agent-100k/']
    comparison(filespaths, 'comparison_100k.png')

    filespaths = ['logdir/a2c-1M/', 'logdir/ppo_attention-1M/', 'logdir/ppo-1M/', 'logdir/random_agent-1M/']
    comparison(filespaths)
    compare_all_success_rates(filespaths)

