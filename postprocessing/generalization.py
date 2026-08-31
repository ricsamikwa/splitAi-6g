"""
robustness_eval.py

Evaluates the generalization of trained DRL split-inference policies (DDQN, A2C, PPO) against OPT on a
held-out dataset the models were NOT trained on.

OPT and the DRL algorithms were run over DIFFERENT NUMBERS of episodes on the new dataset - OPT over 9
episodes, each DRL algorithm over 99 - and use different CSV filename conventions (OPT: "inference_time_
1.csv", unpadded; DRL: "inference_time_01.csv", zero-padded to 2 digits). Because the episode counts don't
match, there is no natural step-for-step pairing between a specific OPT decision and a specific DRL
decision the way there would be if both had run the identical episode sequence - so this version compares
each algorithm's AGGREGATE (mean) value per KPI against OPT's aggregate mean, rather than averaging
per-step percentage errors. This is a genuinely different quantity from a strict per-step MAPE (mean of
|error_i|/|ref_i| across matched steps i) - it's the percentage difference between two overall averages,
each computed over its own (differently-sized) sample. Both OPT's and each DRL algorithm's KPIs are still
read from PRE-COMPUTED CSV logs - this script does not run any agent itself.

MAPE is computed separately for each of: inference_time, ue_energy_comp, ue_energy_comm, and
top1_accuracy_confidence - NOT combined into a single "ue_energy" figure, matching how comp/comm are
reported as separate series everywhere else in this project's charts.

Column format CONFIRMED from earlier example files: all four KPIs use 'time_step' as the x-column.

PLACEHOLDERS - fill in before running:
  - OPT_LOG_FOLDER: where OPT's re-run logs for the new dataset live.
  - DRL_LOG_FOLDERS: where each DRL algorithm's feedforward-generated logs for the new dataset live.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.logging_utils import read_single_col_data

# PLACEHOLDER - point this at wherever OPT's re-run KPI logs for the new dataset are saved.
OPT_LOG_FOLDER = 'optimum'

# PLACEHOLDER - point each entry at wherever that algorithm's feedforward-generated KPI logs for the new
# dataset are saved. Add/remove keys here if you're evaluating a different set of algorithms.
DRL_LOG_FOLDERS = {
    'ddqn': 'rl/ddqn',
    'a2c': 'rl/a2c',
    'ppo': 'rl/ppo',
}

# OPT: 9 episodes, filenames unpadded (e.g. "inference_time_1.csv" ... "inference_time_9.csv").
OPT_EPISODE_NUMBERS = list(range(1, 10))
# DRL: 99 episodes, filenames zero-padded to 2 digits (e.g. "inference_time_01.csv" ... "..._99.csv").
DRL_EPISODE_NUMBERS = list(range(1, 100))

# Each DRL algorithm's IN-DISTRIBUTION (training-distribution) aggregate KPI values, at the fixed Z=2.0s
# deadline - transcribed from Fig. 6 ("performance_split_algorithms_all"), the same aggregate comparison
# chart used earlier for the Fig. 1/3/6 discussion. These are the reference points evaluate_generalization
# _gap() compares each algorithm's held-out-dataset behavior against, to isolate how much a given policy's
# OWN behavior changes on new data - as distinct from evaluate_robustness()'s OPT-referenced comparison,
# which instead measures how far from optimal DRL is on the new dataset (conflating approximation quality
# with genuine distribution-shift degradation - see this file's module docstring for the full rationale).
# CAUTION: manually transcribed from the chart's printed bar labels, not read from underlying log files -
# replace with an exact re-read from whatever produced Fig. 6 if precision beyond ~3 decimal places matters.
DRL_TRAINING_REFERENCE = {
    'ddqn': {'inference_time': 0.415, 'ue_energy_comp': 0.696, 'ue_energy_comm': 0.160, 'top1': 0.568},
    'a2c':  {'inference_time': 0.470, 'ue_energy_comp': 1.026, 'ue_energy_comm': 0.135, 'top1': 0.553},
    'ppo':  {'inference_time': 0.418, 'ue_energy_comp': 0.710, 'ue_energy_comm': 0.134, 'top1': 0.595},
}

# PLACEHOLDER - path to a results CSV, in the same shape evaluate_robustness() saves (index = algorithm
# name; columns include mean_<kpi> for each KPI), computed on a DIFFERENT dataset drawn from the SAME
# distribution as training (not a distribution-shift dataset - see evaluate_generalization_gap_from_
# results_csv()'s docstring). Fill in before running.
DIFFERENT_DATASET_MAPE_RESULTS_CSV = 'logs/robustness/mape_results.csv'

# (csv_x_column, dict_key_used_when_the_csv_was_written) per KPI.
KPI_COLUMN_NAMES = {
    'inference_time': ('time_step', 'inference_time'),
    'ue_energy_comp': ('time_step', 'ue_energy_comp'),
    'ue_energy_comm': ('time_step', 'ue_energy_comm'),
    'top1': ('time_step', 'top1'),
}

# Display settings for the MAPE bar chart - kept consistent with this project's established palette
# (dark blue=ddqn, medium blue=a2c, gray=ppo) so this figure matches the rest of the manuscript's charts.
ALGO_COLORS = {
    'ddqn': '#114584',
    'a2c': '#165DB1',
    'ppo': '#475058',
}
ALGO_ORDER = ['ddqn', 'a2c', 'ppo']
MAPE_METRIC_ORDER = ['mape_inference_time', 'mape_ue_energy_comp', 'mape_ue_energy_comm', 'mape_accuracy']
MAPE_METRIC_LABELS = {
    'mape_inference_time': 'Inference\ntime',
    'mape_ue_energy_comp': 'UE energy\n(comp)',
    'mape_ue_energy_comm': 'UE energy\n(comm)',
    'mape_accuracy': 'Top-1\naccuracy',
}
SMAPE_METRIC_ORDER = ['smape_inference_time', 'smape_ue_energy_comp', 'smape_ue_energy_comm', 'smape_accuracy']
SMAPE_METRIC_LABELS = {
    'smape_inference_time': 'Inference\ntime',
    'smape_ue_energy_comp': 'UE energy\n(comp)',
    'smape_ue_energy_comm': 'UE energy\n(comm)',
    'smape_accuracy': 'Top-1\naccuracy',
}
# Same idea, for evaluate_generalization_gap() (DRL's training-distribution value vs. its own held-out
# -dataset value) rather than evaluate_robustness() (OPT vs. DRL on the held-out dataset).
TRAIN_MAPE_METRIC_ORDER = ['train_mape_inference_time', 'train_mape_ue_energy_comp',
                           'train_mape_ue_energy_comm', 'train_mape_accuracy']
TRAIN_MAPE_METRIC_LABELS = {
    'train_mape_inference_time': 'Inference\ntime',
    'train_mape_ue_energy_comp': 'UE energy\n(comp)',
    'train_mape_ue_energy_comm': 'UE energy\n(comm)',
    'train_mape_accuracy': 'Top-1\naccuracy',
}
TRAIN_SMAPE_METRIC_ORDER = ['train_smape_inference_time', 'train_smape_ue_energy_comp',
                            'train_smape_ue_energy_comm', 'train_smape_accuracy']
TRAIN_SMAPE_METRIC_LABELS = {
    'train_smape_inference_time': 'Inference\ntime',
    'train_smape_ue_energy_comp': 'UE energy\n(comp)',
    'train_smape_ue_energy_comm': 'UE energy\n(comm)',
    'train_smape_accuracy': 'Top-1\naccuracy',
}


# ---------------------------------------------------------------------------
# Aggregate percentage error
# ---------------------------------------------------------------------------
def aggregate_mape(reference, predicted):
    """
    Percentage difference between the AGGREGATE MEAN of `reference` (OPT) and the aggregate mean of
    `predicted` (a DRL algorithm) - NOT a per-step MAPE. This is the right comparison when the two arrays
    come from different numbers of episodes (here: OPT's 9 vs. each DRL algorithm's 99) and therefore
    cannot be paired step-for-step - each array is reduced to a single summary value (its mean) first,
    and the percentage error is computed between those two summary values.

    Args:
        reference (array-like): OPT's values for this KPI, across all of OPT's episodes.
        predicted (array-like): The DRL algorithm's values for this KPI, across all of ITS episodes -
            does NOT need to be the same length as `reference`.

    Returns:
        float: percentage difference between mean(predicted) and mean(reference), i.e.
            100 * |mean(reference) - mean(predicted)| / |mean(reference)|.
    """
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ref_mean = float(np.mean(reference))
    pred_mean = float(np.mean(predicted))
    if ref_mean == 0:
        raise ValueError("OPT's aggregate mean for this KPI is 0 - percentage error is undefined.")
    return 100.0 * abs(ref_mean - pred_mean) / abs(ref_mean)


def aggregate_smape(reference, predicted):
    """
    Symmetric aggregate percentage error between the aggregate mean of `reference` (OPT) and the
    aggregate mean of `predicted` (a DRL algorithm) - the aggregate-level counterpart to aggregate_mape(),
    dividing by the AVERAGE of the two magnitudes instead of just the reference. This keeps a KPI whose
    OPT-side mean happens to sit at a much smaller scale than the others (e.g. ue_energy_comp, where OPT's
    mean can be much smaller than DDQN's) from producing a disproportionately large percentage purely
    because of how small the reference happens to be, rather than because the absolute gap is actually
    larger than for other KPIs - see aggregate_mape()'s docstring for the corresponding non-symmetric
    version and evaluate_robustness()'s docstring for why aggregation is used at all.

    NOTE: this is NOT the standard per-observation sMAPE formula (sMAPE = 100%/n * sum_t |A_t - F_t| /
    ((|A_t| + |F_t|)/2)), which requires PAIRED observations at matching indices t - OPT's 9 episodes and
    each DRL algorithm's 99 have no such correspondence. This instead applies the same symmetric-
    denominator idea to the two AGGREGATE MEANS directly, computed once rather than averaged over paired
    per-step values. Report/cite it as an aggregate-level adaptation of sMAPE, not the textbook formula.

    Args:
        reference (array-like): OPT's values for this KPI, across all of OPT's episodes.
        predicted (array-like): The DRL algorithm's values for this KPI, across all of ITS episodes -
            does NOT need to be the same length as `reference`.

    Returns:
        float: 200 * |mean(reference) - mean(predicted)| / (|mean(reference)| + |mean(predicted)|).
            Bounded in [0, 200] by construction, unlike aggregate_mape() which is unbounded above.
    """
    reference = np.asarray(reference, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ref_mean = float(np.mean(reference))
    pred_mean = float(np.mean(predicted))
    denom = abs(ref_mean) + abs(pred_mean)
    if denom == 0:
        raise ValueError("Both OPT's and the DRL algorithm's aggregate means for this KPI are 0 - "
                          "percentage error is undefined.")
    return 200.0 * abs(ref_mean - pred_mean) / denom


# ---------------------------------------------------------------------------
# KPI loading - shared by OPT and every DRL algorithm, since both are now pre-computed CSVs in the same
# logs/<folder>/system/<metric>_<episode>.csv format (this used to be OPT-only as load_opt_kpis(); DRL's
# logs no longer need live agent execution to produce, since you've already generated them via feedforward,
# so the same loader now serves both roles).
# ---------------------------------------------------------------------------
def _check_kpi_file_exists(base_path):
    """
    Checks whether base_path (with or without a '.csv' suffix - read_single_col_data's exact convention
    isn't visible from here) resolves to a real file, and raises a clear, diagnostic FileNotFoundError if
    not: showing the current working directory and the absolute path that was actually checked, rather
    than the bare relative-path error read_single_col_data itself would raise. Relative paths like
    'logs/<folder>/system/<file>.csv' only resolve correctly when the script is run from the project's
    root directory - running it from anywhere else produces exactly this kind of not-found error even
    when the file genuinely exists on disk.

    Args:
        base_path (str): the path (without a guaranteed '.csv' suffix) that will be passed to
            read_single_col_data.
    """
    candidates = [base_path, base_path + '.csv']
    if any(os.path.exists(c) for c in candidates):
        return
    raise FileNotFoundError(
        f"Could not find '{base_path}' (checked both with and without a '.csv' suffix).\n"
        f"  Current working directory: {os.getcwd()}\n"
        f"  Absolute path checked: {os.path.abspath(base_path)}\n"
        f"If the file genuinely exists elsewhere on disk, this is very likely a working-directory "
        f"mismatch - relative paths like 'logs/<folder>/system/...' only resolve correctly when this "
        f"script is run from the project's root directory (the one containing 'logs/' as a direct "
        f"subfolder). Either run the script from there, or check that OPT_LOG_FOLDER/DRL_LOG_FOLDERS "
        f"point at the right place relative to wherever you're actually running from.")


def load_kpis_from_folder(episode_numbers, log_folder, episode_num_formatter=str):
    """
    Reads pre-computed per-step KPI logs for the given episodes from log_folder, in the
    logs/<folder>/system/<metric>_<episode>.csv format write_logs() (utils/logging_utils.py) produces.
    Concatenates across episodes IN THE GIVEN ORDER.

    Args:
        episode_numbers (list[int]): the (logged) episode numbers to read and concatenate.
        log_folder (str): folder this algorithm's CSVs live under (relative to 'logs/', matching
            writeToCsv's own 'logs/{folder}/{filename}.csv' convention).
        episode_num_formatter (callable): converts an int episode number into the string used in the
            filename. Defaults to str (unpadded, e.g. 1 -> "1", matching OPT's "inference_time_1.csv").
            Pass e.g. `lambda ep: f'{ep:02d}'` for DRL's zero-padded convention ("inference_time_01.csv").

    Returns:
        dict with keys 'inference_time', 'ue_energy_comp', 'ue_energy_comm', 'top1', each a flat numpy
        array concatenated across all given episodes in order.
    """
    all_vals = {kpi: [] for kpi in KPI_COLUMN_NAMES}
    for ep in episode_numbers:
        ep_str = episode_num_formatter(ep)
        per_kpi_this_episode = {}
        for kpi, (x_col, y_col) in KPI_COLUMN_NAMES.items():
            base_path = f'logs/{log_folder}/system/{kpi}_{ep_str}'
            _check_kpi_file_exists(base_path)
            _, vals = read_single_col_data(base_path, x_col, y_col, float, float)
            per_kpi_this_episode[kpi] = vals

        lengths = {kpi: len(v) for kpi, v in per_kpi_this_episode.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Logs in '{log_folder}' for episode {ep_str} have mismatched step counts across KPI "
                f"files ({lengths}) - can't pair them step-for-step.")

        for kpi, vals in per_kpi_this_episode.items():
            all_vals[kpi].extend(vals)

    return {kpi: np.asarray(vals, dtype=float) for kpi, vals in all_vals.items()}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def evaluate_robustness(opt_episode_numbers=OPT_EPISODE_NUMBERS, drl_episode_numbers=DRL_EPISODE_NUMBERS,
                         opt_log_folder=OPT_LOG_FOLDER, drl_log_folders=DRL_LOG_FOLDERS):
    """
    Loads OPT's KPIs (over opt_episode_numbers, unpadded filenames) and every DRL algorithm's KPIs (over
    drl_episode_numbers, zero-padded filenames), and computes the aggregate percentage error (OPT's mean
    as reference) per algorithm per KPI - see aggregate_mape()'s docstring for why this, rather than a
    per-step MAPE, is used when the two sides come from different numbers of episodes.

    Also stores the raw aggregate means (OPT's and the DRL algorithm's) alongside each percentage error -
    aggregate_mape()'s denominator is OPT's mean, so a small denominator can inflate the percentage even
    when the absolute gap is small; having the raw means next to the percentage lets that be checked
    directly instead of guessing from the percentage alone. Also computes aggregate_smape() alongside
    aggregate_mape() for each KPI - see aggregate_smape()'s docstring for why (it avoids the same
    small-denominator distortion that can make one KPI's bar dwarf the others purely due to scale).

    Args:
        opt_episode_numbers (list[int]): OPT's episode numbers. Defaults to OPT_EPISODE_NUMBERS (1-9).
        drl_episode_numbers (list[int]): each DRL algorithm's episode numbers. Defaults to
            DRL_EPISODE_NUMBERS (1-99).
        opt_log_folder (str): folder OPT's logs live under. Defaults to OPT_LOG_FOLDER.
        drl_log_folders (dict[str, str]): {algorithm_name: log_folder} for each DRL algorithm to evaluate.
            Defaults to DRL_LOG_FOLDERS.

    Returns:
        pandas.DataFrame indexed by algorithm name. For each KPI (inference_time, ue_energy_comp,
        ue_energy_comm, accuracy): a mape_<kpi> column and a smape_<kpi> column (the two percentage error
        variants - see aggregate_mape() and aggregate_smape()), a mean_opt_<kpi> column (OPT's aggregate
        mean - identical across every algorithm's row, since it's the shared reference), and a mean_<kpi>
        column (that algorithm's own aggregate mean).
    """
    print(f"Loading OPT's pre-computed KPIs from logs/{opt_log_folder}/system/ for episodes "
          f"{opt_episode_numbers}...")
    opt_kpis = load_kpis_from_folder(opt_episode_numbers, log_folder=opt_log_folder, episode_num_formatter=str)

    # KPI dict key -> the suffix used for this KPI's columns in the results table (top1's suffix is
    # 'accuracy' to match the existing mape_accuracy naming already used elsewhere in this script/plot).
    kpi_to_suffix = {
        'inference_time': 'inference_time',
        'ue_energy_comp': 'ue_energy_comp',
        'ue_energy_comm': 'ue_energy_comm',
        'top1': 'accuracy',
    }

    results = {}
    for agent_type, log_folder in drl_log_folders.items():
        print(f"Loading {agent_type.upper()}'s pre-computed KPIs from logs/{log_folder}/system/ for "
              f"episodes {drl_episode_numbers[0]}-{drl_episode_numbers[-1]}...")
        drl_kpis = load_kpis_from_folder(drl_episode_numbers, log_folder=log_folder,
                                          episode_num_formatter=lambda ep: f'{ep:02d}')

        algo_results = {}
        for kpi_key, suffix in kpi_to_suffix.items():
            opt_mean = float(np.mean(opt_kpis[kpi_key]))
            drl_mean = float(np.mean(drl_kpis[kpi_key]))
            pct_error = aggregate_mape(opt_kpis[kpi_key], drl_kpis[kpi_key])
            sym_pct_error = aggregate_smape(opt_kpis[kpi_key], drl_kpis[kpi_key])
            print(f"  [{agent_type}] {kpi_key}: mean(OPT)={opt_mean:.4g}, mean({agent_type})={drl_mean:.4g}, "
                  f"mape={pct_error:.1f}%, smape={sym_pct_error:.1f}%")
            algo_results[f'mean_opt_{suffix}'] = opt_mean
            algo_results[f'mean_{suffix}'] = drl_mean
            algo_results[f'mape_{suffix}'] = pct_error
            algo_results[f'smape_{suffix}'] = sym_pct_error

        results[agent_type] = algo_results

    return pd.DataFrame(results).T


def evaluate_generalization_gap(drl_episode_numbers=DRL_EPISODE_NUMBERS, drl_log_folders=DRL_LOG_FOLDERS,
                                 training_reference=DRL_TRAINING_REFERENCE):
    """
    For each DRL algorithm, compares its held-out-dataset aggregate KPI means against its OWN training-
    distribution values (training_reference, transcribed from Fig. 6 at Z=2.0s by default) - this is the
    standard train-vs-test "generalization gap" comparison (same model, same metric, training conditions
    vs. unseen conditions), as opposed to evaluate_robustness()'s OPT-referenced comparison, which measures
    how far from OPTIMAL a policy is on new data rather than how much ITS OWN behavior changed getting
    there. The two are complementary, not interchangeable - evaluate_robustness() conflates a policy's
    baseline approximation gap (present even on data it trained on) with genuine distribution-shift
    degradation; this function isolates the second by removing OPT from the comparison entirely.

    Args:
        drl_episode_numbers (list[int]): each DRL algorithm's held-out-dataset episode numbers. Defaults
            to DRL_EPISODE_NUMBERS (1-99).
        drl_log_folders (dict[str, str]): {algorithm_name: log_folder} for each DRL algorithm to evaluate.
            Defaults to DRL_LOG_FOLDERS.
        training_reference (dict[str, dict[str, float]]): {algorithm_name: {kpi: value}} in-distribution
            reference values. Defaults to DRL_TRAINING_REFERENCE - only algorithms present in BOTH this
            dict and drl_log_folders are evaluated (a KeyError is raised otherwise, naming the missing
            algorithm, rather than silently skipping it).

    Returns:
        pandas.DataFrame indexed by algorithm name. For each KPI: a train_mape_<kpi> and train_smape_<kpi>
        column (percentage error vs. that algorithm's OWN training-distribution value), a
        mean_train_<kpi> column (the training-distribution reference value itself), and a mean_new_<kpi>
        column (the held-out-dataset aggregate mean).
    """
    kpi_to_suffix = {
        'inference_time': 'inference_time',
        'ue_energy_comp': 'ue_energy_comp',
        'ue_energy_comm': 'ue_energy_comm',
        'top1': 'accuracy',
    }

    results = {}
    for agent_type, log_folder in drl_log_folders.items():
        if agent_type not in training_reference:
            raise KeyError(
                f"No training-distribution reference values for '{agent_type}' in training_reference - "
                f"add an entry to DRL_TRAINING_REFERENCE (or pass a training_reference dict that includes "
                f"it) before evaluating this algorithm's generalization gap.")

        print(f"Loading {agent_type.upper()}'s pre-computed KPIs from logs/{log_folder}/system/ for "
              f"episodes {drl_episode_numbers[0]}-{drl_episode_numbers[-1]}...")
        drl_kpis = load_kpis_from_folder(drl_episode_numbers, log_folder=log_folder,
                                          episode_num_formatter=lambda ep: f'{ep:02d}')

        algo_results = {}
        for kpi_key, suffix in kpi_to_suffix.items():
            train_value = training_reference[agent_type][kpi_key]
            new_mean = float(np.mean(drl_kpis[kpi_key]))
            train_ref = np.array([train_value])
            pct_error = aggregate_mape(train_ref, drl_kpis[kpi_key])
            sym_pct_error = aggregate_smape(train_ref, drl_kpis[kpi_key])
            print(f"  [{agent_type}] {kpi_key}: train={train_value:.4g}, new={new_mean:.4g}, "
                  f"mape={pct_error:.1f}%, smape={sym_pct_error:.1f}%")
            algo_results[f'mean_train_{suffix}'] = train_value
            algo_results[f'mean_new_{suffix}'] = new_mean
            algo_results[f'train_mape_{suffix}'] = pct_error
            algo_results[f'train_smape_{suffix}'] = sym_pct_error

        results[agent_type] = algo_results

    return pd.DataFrame(results).T


def evaluate_generalization_gap_from_results_csv(results_csv_path=DIFFERENT_DATASET_MAPE_RESULTS_CSV,
                                                  training_reference=DRL_TRAINING_REFERENCE):
    """
    Alternative to evaluate_generalization_gap() that reads each DRL algorithm's aggregate KPI means
    directly from an already-computed results CSV - in the same shape evaluate_robustness() saves (index
    = algorithm name, columns include mean_<kpi> per KPI) - rather than recomputing them from raw
    per-step logs via load_kpis_from_folder(). Use this when you already have a results CSV for a
    different dataset (e.g. produced by running evaluate_robustness() against that dataset and saving its
    output) rather than the raw per-step logs themselves.

    Compares those 'new dataset' means against training_reference (Fig. 6, Z=2.0s - see DRL_TRAINING_
    REFERENCE) exactly as evaluate_generalization_gap() does, returning a DataFrame in the same
    mean_train_*/mean_new_*/train_mape_*/train_smape_* shape - so plot_train_mape_bar_chart() and
    plot_train_smape_bar_chart() work on its output unchanged, without needing separate plotting logic.

    Args:
        results_csv_path (str): path to the results CSV. Defaults to DIFFERENT_DATASET_MAPE_RESULTS_CSV.
        training_reference (dict[str, dict[str, float]]): {algorithm_name: {kpi: value}} in-distribution
            reference values. Defaults to DRL_TRAINING_REFERENCE - only algorithms present in BOTH this
            dict and the CSV's index are evaluated (a KeyError is raised otherwise, naming the missing
            algorithm, rather than silently skipping it).

    Returns:
        pandas.DataFrame indexed by algorithm name, in the same shape evaluate_generalization_gap() returns.
    """
    if results_csv_path == 'PLACEHOLDER_FILL_IN_DIFFERENT_DATASET_MAPE_RESULTS_CSV_PATH':
        raise ValueError(
            "DIFFERENT_DATASET_MAPE_RESULTS_CSV is still the placeholder value - set it near the top of "
            "this file to the path of the different-dataset results CSV before running.")
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(
            f"Could not find '{results_csv_path}'.\n"
            f"  Current working directory: {os.getcwd()}\n"
            f"  Absolute path checked: {os.path.abspath(results_csv_path)}")

    new_results_df = pd.read_csv(results_csv_path, index_col=0)
    print(f"Loaded results for {list(new_results_df.index)} from '{results_csv_path}'")

    kpi_to_suffix = {
        'inference_time': 'inference_time',
        'ue_energy_comp': 'ue_energy_comp',
        'ue_energy_comm': 'ue_energy_comm',
        'top1': 'accuracy',
    }

    results = {}
    for agent_type in new_results_df.index:
        if agent_type not in training_reference:
            raise KeyError(
                f"No training-distribution reference values for '{agent_type}' in training_reference - "
                f"add an entry to DRL_TRAINING_REFERENCE before evaluating this algorithm's generalization gap.")

        algo_results = {}
        for kpi_key, suffix in kpi_to_suffix.items():
            train_value = training_reference[agent_type][kpi_key]
            new_value = float(new_results_df.loc[agent_type, f'mean_{suffix}'])
            pct_error = aggregate_mape(np.array([train_value]), np.array([new_value]))
            sym_pct_error = aggregate_smape(np.array([train_value]), np.array([new_value]))
            print(f"  [{agent_type}] {kpi_key}: train={train_value:.4g}, new={new_value:.4g}, "
                  f"mape={pct_error:.1f}%, smape={sym_pct_error:.1f}%")
            algo_results[f'mean_train_{suffix}'] = train_value
            algo_results[f'mean_new_{suffix}'] = new_value
            algo_results[f'train_mape_{suffix}'] = pct_error
            algo_results[f'train_smape_{suffix}'] = sym_pct_error

        results[agent_type] = algo_results

    return pd.DataFrame(results).T


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_error_bar_chart(df, metric_order, metric_labels, ylabel, title, output_path):
    """
    Shared implementation behind plot_mape_bar_chart() and plot_smape_bar_chart() - plots a grouped bar
    chart of whichever error metric's columns (metric_order) are given, one bar per DRL algorithm per KPI
    group, using this project's established color palette. OPT itself is not plotted - it's the reference
    every value is computed against, so its own error would trivially be 0.

    Args:
        df (pandas.DataFrame): output of evaluate_robustness() - indexed by algorithm name.
        metric_order (list[str]): which of df's columns to plot, in x-axis order (MAPE_METRIC_ORDER or
            SMAPE_METRIC_ORDER).
        metric_labels (dict[str, str]): column name -> x-axis tick label (MAPE_METRIC_LABELS or
            SMAPE_METRIC_LABELS).
        ylabel (str): y-axis label.
        title (str): plot title.
        output_path (str): where to save the figure. Parent folder is created if it doesn't exist.

    Returns:
        None. Saves the figure to output_path.
    """
    algos_present = [a for a in ALGO_ORDER if a in df.index]
    n_algos = len(algos_present)
    n_metrics = len(metric_order)

    x = np.arange(n_metrics)
    bar_width = 0.8 / n_algos

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, algo in enumerate(algos_present):
        values = [df.loc[algo, metric] for metric in metric_order]
        offset = (i - (n_algos - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, width=bar_width, label=algo, color=ALGO_COLORS.get(algo))
        ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metric_order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title='Algorithm', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart to {output_path}")


def plot_mape_bar_chart(df, output_path='logs/robustness/mape_generalization.png'):
    """
    Plots the aggregate MAPE (percentage difference between OPT's and each DRL algorithm's aggregate
    mean per KPI) - see aggregate_mape()'s docstring for what this metric is, and note that a KPI whose
    OPT-side mean sits at a much smaller scale than the others can show a disproportionately large bar
    here even when its absolute gap isn't actually the largest - see plot_smape_bar_chart() for a variant
    that corrects for that.

    Args:
        df (pandas.DataFrame): output of evaluate_robustness() - indexed by algorithm name, with the
            mape_* columns.
        output_path (str): where to save the figure. Parent folder is created if it doesn't exist.

    Returns:
        None. Saves the figure to output_path.
    """
    _plot_error_bar_chart(
        df, MAPE_METRIC_ORDER, MAPE_METRIC_LABELS,
        ylabel='Aggregate % error (MAPE) vs. OPT',
        title='Generalization to held-out dataset: aggregate-mean % error (MAPE) relative to OPT',
        output_path=output_path)


def plot_smape_bar_chart(df, output_path='logs/robustness/smape_generalization.png'):
    """
    Plots the aggregate sMAPE (symmetric percentage difference between OPT's and each DRL algorithm's
    aggregate mean per KPI, dividing by the average of the two magnitudes instead of just OPT's) - see
    aggregate_smape()'s docstring for what this metric is and how it differs from the standard per-
    observation sMAPE formula. Use this alongside (or instead of) plot_mape_bar_chart() when KPIs sit at
    very different scales and you don't want one KPI's bar to visually dwarf the others purely because of
    how small OPT's reference value happens to be for that KPI.

    Args:
        df (pandas.DataFrame): output of evaluate_robustness() - indexed by algorithm name, with the
            smape_* columns.
        output_path (str): where to save the figure. Parent folder is created if it doesn't exist.

    Returns:
        None. Saves the figure to output_path.
    """
    _plot_error_bar_chart(
        df, SMAPE_METRIC_ORDER, SMAPE_METRIC_LABELS,
        ylabel='Symmetric aggregate % error (sMAPE) vs. OPT',
        title='Generalization to held-out dataset: symmetric aggregate % error (sMAPE) relative to OPT',
        output_path=output_path)


def plot_train_mape_bar_chart(df, output_path='logs/robustness/train_mape_generalization_gap.png'):
    """
    Plots the aggregate MAPE between each DRL algorithm's training-distribution value (Fig. 6, Z=2.0s) and
    its own held-out-dataset value - the standard train-vs-test generalization gap, as opposed to
    plot_mape_bar_chart()'s OPT-referenced comparison. See evaluate_generalization_gap()'s docstring.

    Args:
        df (pandas.DataFrame): output of evaluate_generalization_gap() - indexed by algorithm name, with
            the train_mape_* columns.
        output_path (str): where to save the figure. Parent folder is created if it doesn't exist.

    Returns:
        None. Saves the figure to output_path.
    """
    _plot_error_bar_chart(
        df, TRAIN_MAPE_METRIC_ORDER, TRAIN_MAPE_METRIC_LABELS,
        ylabel='Generalization gap (MAPE, %)',
        title='Generalization gap: training-distribution vs. held-out-dataset % error (MAPE)',
        output_path=output_path)


def plot_train_smape_bar_chart(df, output_path='logs/robustness/train_smape_generalization_gap.svg'):
    """
    Symmetric-error (sMAPE) variant of plot_train_mape_bar_chart() - see aggregate_smape()'s docstring
    for why this may be preferable when KPIs sit at different scales.

    Args:
        df (pandas.DataFrame): output of evaluate_generalization_gap() or
            evaluate_generalization_gap_from_results_csv() - indexed by algorithm name, with the
            train_smape_* columns.
        output_path (str): where to save the figure. Parent folder is created if it doesn't exist.

    Returns:
        None. Saves the figure to output_path.
    """
    _plot_error_bar_chart(
        df, TRAIN_SMAPE_METRIC_ORDER, TRAIN_SMAPE_METRIC_LABELS,
        ylabel='Generalization gap (sMAPE, %)',
        title='Generalization gap: training-distribution vs. different dataset (same distribution), sMAPE',
        output_path=output_path)


if __name__ == '__main__':
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    parser = argparse.ArgumentParser(
        description='Evaluate DRL generalization on a held-out dataset, both vs. OPT and vs. each '
                    'algorithm\'s own training-distribution performance.')
    parser.add_argument('--algorithms', nargs='+', default=['ddqn'], choices=list(DRL_LOG_FOLDERS.keys()),
                         help="Which DRL algorithm(s) to evaluate/plot (default: ddqn only). "
                              "e.g. --algorithms ddqn a2c to run more than one.")
    args = parser.parse_args()

    selected_drl_log_folders = {algo: DRL_LOG_FOLDERS[algo] for algo in args.algorithms}
    os.makedirs('logs/robustness', exist_ok=True)

    # OPT_EPISODE_NUMBERS (1-9) and DRL_EPISODE_NUMBERS (1-99) are set near the top of this file - adjust
    # there if your actual episode ranges differ.

    # print("=== Evaluating vs. OPT (re-solved on the held-out dataset) ===")
    # df_opt = evaluate_robustness(drl_log_folders=selected_drl_log_folders)
    # print(df_opt)
    # df_opt.to_csv('logs/robustness/mape_results.csv')
    # plot_mape_bar_chart(df_opt)
    # plot_smape_bar_chart(df_opt)

    print("\n=== Evaluating generalization gap (training-distribution vs. different dataset, same "
          "distribution) ===")
    df_gap = evaluate_generalization_gap_from_results_csv()
    print(df_gap)
    df_gap.to_csv('logs/robustness/generalization_gap_results.csv')
    plot_train_smape_bar_chart(df_gap)