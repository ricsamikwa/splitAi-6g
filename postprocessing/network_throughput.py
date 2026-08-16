import os, csv
import pandas as pd
import numpy as np
from postprocessing.plot_system_kpis import return_order, parse_episode_number
import matplotlib.pyplot as plt


def return_ci_95(sd):
    return 1.96 * sd

def return_ci_90(sd):
    return 1.645 * sd

def read_kpis_from_files(folder, kpi_type, episode_count, max_throughput):
    """
    Function to read system KPIs (inference time, ue computation and communication energy) for each episode.

    Args:
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        kpi_type (str): The kpi to read e.g. inference_time
        episode_count (int): The episode number
        max_throughput (float): the max network throughput of the desired kpi
    Returns:
        Tuple: (time step when kpi was recorded as list, kpi as list )
    """
    #if inference_deadline is None:
    #    file = 'logs/{}/system/{}_{}.csv'.format(folder, kpi_type, episode_count)
    #else:
    if kpi_type == 'split_idx':
        file = 'logs/{}/comparison/thr_{}/splits/{}_{}.csv'.format(folder, max_throughput, kpi_type, episode_count)
    else:
        file = 'logs/{}/comparison/thr_{}/system/{}_{}.csv'.format(folder, max_throughput, kpi_type, episode_count)
    data_timestep = []
    data_kpi = []
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for k, item in enumerate(reader):
            if k != 0:
                data_timestep.append(int(item[0]))
                data_kpi.append(float(item[1]))
    return data_timestep, data_kpi

def parse_kpis(folder, n_episodes, max_throughput):
    """
    Function to read and parse kpis into a 2D pandas DataFrame.
    Args:
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        n_episodes (int): number of episodes per algorithm to print
        max_throughput (float): the max network throughput setting of the desired kpi

    Returns:
        Tuple: (dataframes containing inference time, ue computation and communication energy and energy credit)
    """
    if folder != 'rl/ddqn':
        order = return_order(n_episodes)
    else:
        order = return_order(n_episodes=1000)
    inference_times_all_episodes = []
    ue_energy_comp_all_episodes = []
    ue_energy_comm_all_episodes = []
    energy_credit_all_episodes = []
    y_net_all_episodes = []
    flops_off_all_episodes = []
    time_steps = []
    split_idx_all_episodes = []

    for episode in range(1, n_episodes + 1):
        episode_count = parse_episode_number(order, episode)
        kpi_type = 'inference_time'
        time_steps, inference_times_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, max_throughput)
        inference_times_all_episodes.append(inference_times_per_episode)
        kpi_type = 'ue_energy_comp'
        time_steps, ue_energy_comp_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, max_throughput)
        ue_energy_comp_all_episodes.append(ue_energy_comp_per_episode)
        kpi_type = 'ue_energy_comm'
        time_steps, ue_energy_comm_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, max_throughput)
        ue_energy_comm_all_episodes.append(ue_energy_comm_per_episode)
        kpi_type = 'energy_credit'
        time_steps, energy_credit_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, max_throughput)
        energy_credit_all_episodes.append(energy_credit_per_episode)
        if folder == 'rl/ddqn':
            kpi_type = 'y_net'
            time_steps, y_net_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, max_throughput)
            y_net_all_episodes.append(y_net_per_episode)
            kpi_type = 'flops_off'
            #time_steps, flops_off_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
            #flops_off_all_episodes.append(flops_off_per_episode)
        if folder == 'heuristic':
            kpi_type = 'split_idx'
            time_steps, split_idx_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, max_throughput)
            split_idx_all_episodes.append(split_idx_per_episode)

    # concatenate data of all episodes into single data structure
    df_inference_time = pd.DataFrame(inference_times_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_ue_energy_comp = pd.DataFrame(ue_energy_comp_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_ue_energy_comm = pd.DataFrame(ue_energy_comm_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_energy_credit = pd.DataFrame(energy_credit_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_y_net = pd.DataFrame(y_net_all_episodes, columns=time_steps,
                                    index=[ep for ep in range(1, n_episodes + 1)])
    #df_flops_off = pd.DataFrame(flops_off_all_episodes, columns=time_steps,
    #                                index=[ep for ep in range(1, n_episodes + 1)])
    if folder == 'heuristic':
        df_split_idx = pd.DataFrame(split_idx_all_episodes,  columns=time_steps,
                                    index=[ep for ep in range(1, n_episodes + 1)])
        return df_inference_time, df_ue_energy_comp, df_ue_energy_comm, df_energy_credit, df_y_net, df_split_idx

    return df_inference_time, df_ue_energy_comp, df_ue_energy_comm, df_energy_credit, df_y_net, None


def _plot_errorbar_series(x_values, y_values, y_err, color, label, offset):
    """
    Draws one algorithm's error-bar series, offsetting its x-values by `offset` so that points for
    different algorithms at the same nominal network-throughput value don't overlap. Consolidating the
    previously near-duplicate errorbar() calls into one helper is what let SERIES_OFFSETS (below) become
    the single source of truth for spacing - the bug this replaces (ppo's offset of +1 vs. every other
    series' 4-8 spacing) was exactly the kind of copy-paste drift six independent inline blocks invite.
    """
    plt.errorbar(x=np.array(x_values) + offset, y=y_values, yerr=y_err,
                 color=color, ecolor=color, elinewidth=2, capsize=4, capthick=2,
                 linestyle='-', fmt='o', label=label)


# Evenly-spaced x-offsets (in MB/s) for each series, symmetric around 0. Computed from the number of
# series rather than hardcoded per-call, so adding/removing a series (e.g. re-enabling 'heuristic') just
# means adding/removing one entry here rather than manually re-balancing every offset by hand.
# Total spread is kept well under the 50 MB/s gap between adjacent throughput values in
# max_network_throughput_list, so one throughput value's jittered cluster never overlaps its neighbor's.
_SERIES_ORDER = ['opt', 'ddqn', 'a2c', 'ppo', 'heuristic_shallow', 'heuristic_deep', 'random', 'fixed']
_JITTER_SPREAD = 15  # +/- MB/s from center; total spread across all series stays within +/-15
_SERIES_OFFSETS = dict(zip(_SERIES_ORDER, np.linspace(-_JITTER_SPREAD, _JITTER_SPREAD, len(_SERIES_ORDER))))


def _plot_errorbar_series(x_values, y_values, y_err, color, label, offset):
    """
    Draws one algorithm's error-bar series, offsetting its x-values by `offset` so that points for
    different algorithms at the same nominal network-throughput value don't overlap. Consolidating the
    previously near-duplicate errorbar() calls into one helper is what let SERIES_OFFSETS (below) become
    the single source of truth for spacing - the bug this replaces (ppo's offset of +1 vs. every other
    series' 4-8 spacing) was exactly the kind of copy-paste drift six independent inline blocks invite.
    """
    plt.errorbar(x=np.array(x_values) + offset, y=y_values, yerr=y_err,
                 color=color, ecolor=color, elinewidth=2, capsize=4, capthick=2,
                 linestyle='-', fmt='o', label=label)


# Evenly-spaced x-offsets (in MB/s) for each series, symmetric around 0. Computed from the number of
# series rather than hardcoded per-call, so adding/removing a series (e.g. re-enabling 'heuristic') just
# means adding/removing one entry here rather than manually re-balancing every offset by hand.
# Total spread is kept well under the 50 MB/s gap between adjacent throughput values in
# max_network_throughput_list, so one throughput value's jittered cluster never overlaps its neighbor's.
_SERIES_ORDER = ['opt', 'ddqn', 'a2c', 'ppo', 'heuristic_shallow', 'heuristic_deep', 'random', 'fixed']
_JITTER_SPREAD = 15  # +/- MB/s from center; total spread across all series stays within +/-15
_SERIES_OFFSETS = dict(zip(_SERIES_ORDER, np.linspace(-_JITTER_SPREAD, _JITTER_SPREAD, len(_SERIES_ORDER))))

# The actual split_idx values logged for the two heuristic candidates - confirmed via
# split_per_episode.value_counts() to be 10.0 and 20.0, NOT 1/2 as originally assumed (split_idx is
# enumerate_action_space()'s global index over the full split action space, not a simple 1st/2nd label).
# UNCONFIRMED: which of these two is actually the shallow ([(0,0,3),...]) vs. deep ([(0,0,6),...]) split -
# swap the two values below if the resulting shallow/deep labels come out looking reversed in the figure.
SPLIT_IDX_SHALLOW = 10
SPLIT_IDX_DEEP = 20


def plot_kpis_vs_max_throughput(df_all_inference_time, df_all_ue_energy_comp, df_all_ue_energy_comm,
                                    max_throughput_list, n_episodes_to_train, total_episodes_train, df_split_idx_list_hr):
    """
    Script to plot the comparison between inference time vs energy credit usage, or ue energy vs energy credit usage.
    Args:
        df_inference_time_list (list of pandas DataFrame): list of 2D dataframes containing of inference time logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_ue_energy_comp_list (list of pandas DataFrame): list of 2D dataframes containing of ue_energy_comp logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_ue_energy_comm_list (list of pandas DataFrame): list of 2D dataframes containing of ue_energy_comm logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        max_throughput_list (list): list of max throughput values to plot.

    Returns:

    """
    #r = np.arange(len(algorithms))  # the label locations
    #width = 0.25  # the width of the bars
    #multiplier = 0
    df_inference_time_ddqn_list = df_all_inference_time[0]   # ddqn
    df_ue_energy_comp_ddqn_list = df_all_ue_energy_comp[0]   # ddqn
    df_ue_energy_comm_ddqn_list = df_all_ue_energy_comm[0]   # ddqn
    df_inference_time_a2c_list = df_all_inference_time[1]   # a2c
    df_ue_energy_comp_a2c_list = df_all_ue_energy_comp[1]   # a2c
    df_ue_energy_comm_a2c_list = df_all_ue_energy_comm[1]   # a2c
    df_inference_time_ppo_list = df_all_inference_time[2]  # ppo
    df_ue_energy_comp_ppo_list = df_all_ue_energy_comp[2]  # ppo
    df_ue_energy_comm_ppo_list = df_all_ue_energy_comm[2]  # ppo
    df_inference_time_opt_list = df_all_inference_time[3]   # opt
    df_ue_energy_comp_opt_list = df_all_ue_energy_comp[3]   # opt
    df_ue_energy_comm_opt_list = df_all_ue_energy_comm[3]   # opt
    df_inference_time_random_list = df_all_inference_time[4]    # random
    df_ue_energy_comp_random_list = df_all_ue_energy_comp[4]    # random
    df_ue_energy_comm_random_list = df_all_ue_energy_comm[4]    # random
    df_inference_time_fixed_list = df_all_inference_time[5]     # fixed
    df_ue_energy_comp_fixed_list = df_all_ue_energy_comp[5]     # fixed
    df_ue_energy_comm_fixed_list = df_all_ue_energy_comm[5]     # fixed
    df_inference_time_hr_list = df_all_inference_time[6]     # heuristic
    df_ue_energy_comp_hr_list = df_all_ue_energy_comp[6]     # heuristic
    df_ue_energy_comm_hr_list = df_all_ue_energy_comm[6]     # heuristic

    n_episodes_aft_train_ddqn = total_episodes_train['rl/ddqn'] - n_episodes_to_train['rl/ddqn']
    n_episodes_aft_train_a2c = total_episodes_train['rl/a2c'] - n_episodes_to_train['rl/a2c']
    n_episodes_aft_train_ppo = total_episodes_train['rl/ppo'] - n_episodes_to_train['rl/ppo']
    fig, ax = plt.subplots()
    inference_time_mean_per_deadline_ddqn = []
    inference_time_sd_per_deadline_ddqn = []
    inference_time_ci_per_deadline_ddqn = []
    ue_energy_comp_mean_per_deadline_ddqn = []
    ue_energy_comm_mean_per_deadline_ddqn = []
    inference_time_mean_per_deadline_a2c = []
    inference_time_sd_per_deadline_a2c = []
    inference_time_ci_per_deadline_a2c = []
    ue_energy_comp_mean_per_deadline_a2c = []
    ue_energy_comm_mean_per_deadline_a2c = []
    inference_time_mean_per_deadline_ppo = []
    inference_time_sd_per_deadline_ppo = []
    inference_time_ci_per_deadline_ppo = []
    ue_energy_comp_mean_per_deadline_ppo = []
    ue_energy_comm_mean_per_deadline_ppo = []
    inference_time_mean_per_deadline_opt = []
    inference_time_sd_per_deadline_opt = []
    inference_time_ci_per_deadline_opt = []
    ue_energy_comp_mean_per_deadline_opt = []
    ue_energy_comm_mean_per_deadline_opt = []
    inference_time_mean_per_deadline_random = []
    inference_time_sd_per_deadline_random = []
    inference_time_ci_per_deadline_random = []
    ue_energy_comp_mean_per_deadline_random = []
    ue_energy_comm_mean_per_deadline_random = []
    inference_time_mean_per_deadline_fixed = []
    inference_time_sd_per_deadline_fixed = []
    inference_time_ci_per_deadline_fixed = []
    ue_energy_comp_mean_per_deadline_fixed = []
    ue_energy_comm_mean_per_deadline_fixed = []
    # heuristic is split into two series: episodes where the run's one-time random split choice landed
    # on the shallow partition (split_idx == 1) vs. the deep partition (split_idx == 2) - see
    # split_generator.py's heuristic(). Pooling both into one mean/CI (the previous version) mixes two
    # structurally different regimes into a single, misleadingly wide, bimodal-looking band.
    inference_time_mean_per_deadline_hr_shallow = []
    inference_time_ci_per_deadline_hr_shallow = []
    ue_energy_comp_mean_per_deadline_hr_shallow = []
    ue_energy_comm_mean_per_deadline_hr_shallow = []
    inference_time_mean_per_deadline_hr_deep = []
    inference_time_ci_per_deadline_hr_deep = []
    ue_energy_comp_mean_per_deadline_hr_deep = []
    ue_energy_comm_mean_per_deadline_hr_deep = []
    #print(df_inference_time_ddqn_list[0])
    # extract data and store means
    for i, max_throughput in enumerate(max_throughput_list):
        df_inference_time_ddqn = df_inference_time_ddqn_list[i]
        df_ue_energy_comp_ddqn = df_ue_energy_comp_ddqn_list[i]
        df_ue_energy_comm_ddqn = df_ue_energy_comm_ddqn_list[i]
        # then calculate the means
        df_inference_time_ddqn['mean'] = df_inference_time_ddqn.mean(axis=1)
        inference_time_mean_per_deadline_ddqn.append(df_inference_time_ddqn['mean'].iloc[n_episodes_aft_train_ddqn:total_episodes_train['rl/ddqn']].mean())
        inference_time_ddqn_sd = df_inference_time_ddqn['mean'].std()
        inference_time_ci_ddqn = return_ci_90(inference_time_ddqn_sd)
        #print(df_inference_time_ddqn)
        #print(df_inference_time_ddqn_sd)
        inference_time_sd_per_deadline_ddqn.append(inference_time_ddqn_sd)
        inference_time_ci_per_deadline_ddqn.append(inference_time_ci_ddqn)
        df_ue_energy_comp_ddqn['mean'] = df_ue_energy_comp_ddqn.mean(axis=1)
        ue_energy_comp_mean_per_deadline_ddqn.append(df_ue_energy_comp_ddqn['mean'].iloc[n_episodes_aft_train_ddqn:total_episodes_train['rl/ddqn']].mean())
        df_ue_energy_comm_ddqn['mean'] = df_ue_energy_comm_ddqn.mean(axis=1)
        ue_energy_comm_mean_per_deadline_ddqn.append(df_ue_energy_comm_ddqn['mean'].iloc[n_episodes_aft_train_ddqn:total_episodes_train['rl/ddqn']].mean())
        # sum_ue_energy_per_deadline.append(df_ue_energy_comp_ddqn['mean'][n_episodes_bef_train:].mean() +
        #                                   df_ue_energy_comm_ddqn['mean'][n_episodes_bef_train:])
        # then a2c
        df_inference_time_a2c = df_inference_time_a2c_list[i]
        df_ue_energy_comp_a2c = df_ue_energy_comp_a2c_list[i]
        df_ue_energy_comm_a2c = df_ue_energy_comm_a2c_list[i]
        # then calculate the means
        df_inference_time_a2c['mean'] = df_inference_time_a2c.mean(axis=1)
        inference_time_mean_per_deadline_a2c.append(
            df_inference_time_a2c['mean'].iloc[n_episodes_aft_train_a2c:total_episodes_train['rl/a2c']].mean())
        inference_time_a2c_sd = df_inference_time_a2c['mean'].std()
        inference_time_ci_a2c = return_ci_90(inference_time_a2c_sd)
        inference_time_sd_per_deadline_a2c.append(inference_time_a2c_sd)
        inference_time_ci_per_deadline_a2c.append(inference_time_ci_a2c)
        df_ue_energy_comp_a2c['mean'] = df_ue_energy_comp_a2c.mean(axis=1)
        ue_energy_comp_mean_per_deadline_a2c.append(
            df_ue_energy_comp_a2c['mean'].iloc[n_episodes_aft_train_a2c:total_episodes_train['rl/a2c']].mean())
        df_ue_energy_comm_a2c['mean'] = df_ue_energy_comm_a2c.mean(axis=1)
        ue_energy_comm_mean_per_deadline_a2c.append(
            df_ue_energy_comm_a2c['mean'].iloc[n_episodes_aft_train_a2c:total_episodes_train['rl/a2c']].mean())

        # then ppo
        df_inference_time_ppo = df_inference_time_ppo_list[i]
        df_ue_energy_comp_ppo = df_ue_energy_comp_ppo_list[i]
        df_ue_energy_comm_ppo = df_ue_energy_comm_ppo_list[i]
        # then calculate the means
        df_inference_time_ppo['mean'] = df_inference_time_ppo.mean(axis=1)
        inference_time_mean_per_deadline_ppo.append(
            df_inference_time_ppo['mean'].iloc[n_episodes_aft_train_ppo:total_episodes_train['rl/ppo']].mean())
        inference_time_ppo_sd = df_inference_time_ppo['mean'].std()
        inference_time_ci_ppo = return_ci_90(inference_time_ppo_sd)
        inference_time_sd_per_deadline_ppo.append(inference_time_ppo_sd)
        inference_time_ci_per_deadline_ppo.append(inference_time_ci_ppo)
        df_ue_energy_comp_ppo['mean'] = df_ue_energy_comp_ppo.mean(axis=1)
        ue_energy_comp_mean_per_deadline_ppo.append(
            df_ue_energy_comp_ppo['mean'].iloc[n_episodes_aft_train_ppo:total_episodes_train['rl/ppo']].mean())
        df_ue_energy_comm_ppo['mean'] = df_ue_energy_comm_ppo.mean(axis=1)
        ue_energy_comm_mean_per_deadline_ppo.append(
            df_ue_energy_comm_ppo['mean'].iloc[n_episodes_aft_train_ppo:total_episodes_train['rl/ppo']].mean())

        # then optimum
        df_inference_time_opt = df_inference_time_opt_list[i]
        df_ue_energy_comp_opt = df_ue_energy_comp_opt_list[i]
        df_ue_energy_comm_opt = df_ue_energy_comm_opt_list[i]
        df_inference_time_opt['mean'] = df_inference_time_opt.mean(axis=1)
        inference_time_mean_per_deadline_opt.append(df_inference_time_opt['mean'].mean())
        inference_time_opt_sd = df_inference_time_opt['mean'].std()
        inference_time_sd_per_deadline_opt.append(inference_time_opt_sd)
        inference_time_opt_ci = return_ci_90(inference_time_opt_sd)
        inference_time_ci_per_deadline_opt.append(inference_time_opt_ci)
        df_ue_energy_comp_opt['mean'] = df_ue_energy_comp_opt.mean(axis=1)
        ue_energy_comp_mean_per_deadline_opt.append(df_ue_energy_comp_opt['mean'].mean())
        df_ue_energy_comm_opt['mean'] = df_ue_energy_comm_opt.mean(axis=1)
        ue_energy_comm_mean_per_deadline_opt.append(df_ue_energy_comm_opt['mean'].mean())

        # then random
        df_inference_time_random = df_inference_time_random_list[i]
        df_ue_energy_comp_random = df_ue_energy_comp_random_list[i]
        df_ue_energy_comm_random = df_ue_energy_comm_random_list[i]
        df_inference_time_random['mean'] = df_inference_time_random.mean(axis=1)
        inference_time_mean_per_deadline_random.append(df_inference_time_random['mean'].mean())
        inference_time_random_sd = df_inference_time_random['mean'].std()
        inference_time_random_ci = return_ci_90(inference_time_random_sd)
        inference_time_sd_per_deadline_random.append(inference_time_random_sd)
        inference_time_ci_per_deadline_random.append(inference_time_random_ci)
        df_ue_energy_comp_random['mean'] = df_ue_energy_comp_random.mean(axis=1)
        ue_energy_comp_mean_per_deadline_random.append(df_ue_energy_comp_random['mean'].mean())
        df_ue_energy_comm_random['mean'] = df_ue_energy_comm_random.mean(axis=1)
        ue_energy_comm_mean_per_deadline_random.append(df_ue_energy_comm_random['mean'].mean())

        # then fixed
        df_inference_time_fixed = df_inference_time_fixed_list[i]
        df_ue_energy_comp_fixed = df_ue_energy_comp_fixed_list[i]
        df_ue_energy_comm_fixed = df_ue_energy_comm_fixed_list[i]
        df_inference_time_fixed['mean'] = df_inference_time_fixed.mean(axis=1)
        inference_time_mean_per_deadline_fixed.append(df_inference_time_fixed['mean'].mean())
        inference_time_fixed_sd = df_inference_time_fixed['mean'].std()
        inference_time_fixed_ci = return_ci_90(inference_time_fixed_sd)
        inference_time_sd_per_deadline_fixed.append(inference_time_fixed_sd)
        inference_time_ci_per_deadline_fixed.append(inference_time_fixed_ci)
        df_ue_energy_comp_fixed['mean'] = df_ue_energy_comp_fixed.mean(axis=1)
        ue_energy_comp_mean_per_deadline_fixed.append(df_ue_energy_comp_fixed['mean'].mean())
        df_ue_energy_comm_fixed['mean'] = df_ue_energy_comm_fixed.mean(axis=1)
        ue_energy_comm_mean_per_deadline_fixed.append(df_ue_energy_comm_fixed['mean'].mean())

        # finally heuristic - split by which of the two candidate splits each episode's one-time random
        # choice landed on (see split_generator.py's heuristic(): split_idx 1 = shallow, 2 = deep). The
        # split choice is made once per episode and held fixed for its duration, so every column within a
        # row should agree - .mode(axis=1) is used (rather than just the first column) as a safety margin
        # against any stray logging artifact, not because the value is actually expected to vary row-wise.
        df_inference_time_hr = df_inference_time_hr_list[i]
        df_ue_energy_comp_hr = df_ue_energy_comp_hr_list[i]
        df_ue_energy_comm_hr = df_ue_energy_comm_hr_list[i]
        df_split_idx_hr = df_split_idx_list_hr[i]

        df_inference_time_hr['mean'] = df_inference_time_hr.mean(axis=1)
        df_ue_energy_comp_hr['mean'] = df_ue_energy_comp_hr.mean(axis=1)
        df_ue_energy_comm_hr['mean'] = df_ue_energy_comm_hr.mean(axis=1)
        split_per_episode = df_split_idx_hr.mode(axis=1)[0]

        mask_shallow = (split_per_episode == SPLIT_IDX_SHALLOW).reindex(df_inference_time_hr.index, fill_value=False)
        mask_deep = (split_per_episode == SPLIT_IDX_DEEP).reindex(df_inference_time_hr.index, fill_value=False)
        n_accounted = int(mask_shallow.sum() + mask_deep.sum())
        if n_accounted == 0:
            raise ValueError(
                f"heuristic split masks matched 0/{len(df_inference_time_hr)} episodes at max_throughput="
                f"{max_throughput} - SPLIT_IDX_SHALLOW/SPLIT_IDX_DEEP ({SPLIT_IDX_SHALLOW}/{SPLIT_IDX_DEEP}) "
                f"don't match the values actually logged (seen: {sorted(split_per_episode.unique())}).")
        elif n_accounted < len(df_inference_time_hr):
            print(f"  [heuristic] warning: only {n_accounted}/{len(df_inference_time_hr)} episodes matched "
                  f"a known split_idx at max_throughput={max_throughput} - "
                  f"unmatched values: {sorted(set(split_per_episode.unique()) - {SPLIT_IDX_SHALLOW, SPLIT_IDX_DEEP})}")

        inference_time_hr_shallow_sd = df_inference_time_hr.loc[mask_shallow, 'mean'].std()
        inference_time_mean_per_deadline_hr_shallow.append(df_inference_time_hr.loc[mask_shallow, 'mean'].mean())
        inference_time_ci_per_deadline_hr_shallow.append(return_ci_90(inference_time_hr_shallow_sd))
        ue_energy_comp_mean_per_deadline_hr_shallow.append(df_ue_energy_comp_hr.loc[mask_shallow, 'mean'].mean())
        ue_energy_comm_mean_per_deadline_hr_shallow.append(df_ue_energy_comm_hr.loc[mask_shallow, 'mean'].mean())

        inference_time_hr_deep_sd = df_inference_time_hr.loc[mask_deep, 'mean'].std()
        inference_time_mean_per_deadline_hr_deep.append(df_inference_time_hr.loc[mask_deep, 'mean'].mean())
        inference_time_ci_per_deadline_hr_deep.append(return_ci_90(inference_time_hr_deep_sd))
        ue_energy_comp_mean_per_deadline_hr_deep.append(df_ue_energy_comp_hr.loc[mask_deep, 'mean'].mean())
        ue_energy_comm_mean_per_deadline_hr_deep.append(df_ue_energy_comm_hr.loc[mask_deep, 'mean'].mean())

        #sum_ue_energy_per_deadline_opt.append(df_inference_time_opt['mean'] + df_ue_energy_comm_opt['mean'])
    # sum_ue_energy_per_deadline_ddqn = np.add(ue_energy_comp_mean_per_deadline_ddqn, ue_energy_comm_mean_per_deadline_ddqn)
    sum_ue_energy_per_deadline_opt = np.add(ue_energy_comp_mean_per_deadline_opt, ue_energy_comm_mean_per_deadline_opt)
    #sum_ue_energy_per_deadline_random = np.add(ue_energy_comp_mean_per_deadline_random, ue_energy_comm_mean_per_deadline_random)
    #sum_ue_energy_per_deadline_fixed = np.add(ue_energy_comp_mean_per_deadline_fixed, ue_energy_comm_mean_per_deadline_fixed)
    #print(sum_ue_energy_per_deadline_ddqn)
    #print(sum_ue_energy_per_deadline_opt)

    # sum energy
    #ax.plot(max_throughput_list, sum_ue_energy_per_deadline_opt, color='#072140', marker='o', label='opt')
    #ax.plot(max_throughput_list, sum_ue_energy_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    #ax.plot(max_throughput_list, sum_ue_energy_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    #ax.plot(max_throughput_list, sum_ue_energy_per_deadline_fixed, color='#8F81EA', marker='o', label='fixed')

    # comp energy
    #ax.plot(max_throughput_list, ue_energy_comp_mean_per_deadline_opt, color='#072140', marker='o', label='opt')
    #ax.plot(max_throughput_list, ue_energy_comp_mean_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    #ax.plot(max_throughput_list, ue_energy_comp_mean_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    #ax.plot(max_throughput_list, ue_energy_comp_mean_per_deadline_fixed, color='#8F81EA', marker='o', label='fixed')

    #print(ue_energy_comm_mean_per_deadline_opt)
    #print(ue_energy_comm_mean_per_deadline_fixed)
    # comm energy
    #ax.plot(max_throughput_list, ue_energy_comm_mean_per_deadline_opt, color='#072140', marker='o', label='opt')
    #ax.plot(max_throughput_list, ue_energy_comm_mean_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    #ax.plot(max_throughput_list, ue_energy_comm_mean_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    #ax.plot(max_throughput_list, ue_energy_comm_mean_per_deadline_fixed, color='#8F81EA', marker='o', label='fixed')
    #ax.plot(inference_deadline_list, sum_ue_energy_per_deadline_opt, marker='o', label='optimum')

    # inference time
    # ax.plot(max_throughput_list, inference_time_mean_per_deadline_opt, color='#072140', marker='o', label='opt')
    # ax.plot(max_throughput_list, inference_time_mean_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    # ax.plot(max_throughput_list, inference_time_mean_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    # #ax.plot(max_throughput_list, inference_time_mean_per_deadline_fixed, color='#8F81EA', marker='o', label='fixed')
    _plot_errorbar_series(max_throughput_list, inference_time_mean_per_deadline_opt,
                           inference_time_ci_per_deadline_opt, "#072140", 'opt', _SERIES_OFFSETS['opt'])
    _plot_errorbar_series(max_throughput_list, inference_time_mean_per_deadline_ddqn,
                           inference_time_ci_per_deadline_ddqn, "#114584", 'ddqn', _SERIES_OFFSETS['ddqn'])
    _plot_errorbar_series(max_throughput_list, inference_time_mean_per_deadline_a2c,
                           inference_time_ci_per_deadline_a2c, "#165DB1", 'a2c', _SERIES_OFFSETS['a2c'])
    _plot_errorbar_series(max_throughput_list, inference_time_mean_per_deadline_ppo,
                           inference_time_ci_per_deadline_ppo, "#475058", 'ppo', _SERIES_OFFSETS['ppo'])
    _plot_errorbar_series(max_throughput_list, inference_time_mean_per_deadline_hr_shallow,
                           inference_time_ci_per_deadline_hr_shallow, "#E36B5B", 'heuristic (shallow)',
                           _SERIES_OFFSETS['heuristic_shallow'])
    _plot_errorbar_series(max_throughput_list, inference_time_mean_per_deadline_hr_deep,
                           inference_time_ci_per_deadline_hr_deep, "#8C2F22", 'heuristic (deep)',
                           _SERIES_OFFSETS['heuristic_deep'])
    _plot_errorbar_series(max_throughput_list, inference_time_mean_per_deadline_random,
                           inference_time_ci_per_deadline_random, "#9ABCE4", 'random', _SERIES_OFFSETS['random'])
    _plot_errorbar_series(max_throughput_list, inference_time_mean_per_deadline_fixed,
                           inference_time_ci_per_deadline_fixed, "#8F81EA", 'fixed', _SERIES_OFFSETS['fixed'])
    ax.set_xlabel('Network throughput (MB/s)')
    ax.set_ylabel('Inference time (s)')
    # #ax.set_ylabel('UE energy comp (J)')
    # #ax.set_ylabel('UE energy comm (J)')
    # #ax.set_ylabel('UE energy sum (J)')
    # #plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.savefig('results/journal/inference_time_vs_max_throughput_all.png')
    plt.savefig('results/journal/inference_time_vs_max_throughput_all.svg')
    plt.show()


def main():
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    max_network_throughput_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]   # in MB/s
    #max_network_throughput_list = [100]
    # specifies the episodes of convergence of ddqn
    n_episodes_to_train = {'rl/ddqn': 3000, 'rl/a2c': 4500, 'rl/ppo': 4500}
    total_episodes_train = {'rl/ddqn': 5000, 'rl/a2c': 5000, 'rl/ppo': 5000}
    df_inference_time_list_ddqn = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_ddqn = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_ddqn = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_a2c = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_a2c = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_a2c = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_ppo = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_ppo = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_ppo = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_opt = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_opt = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_opt = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_random = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_random = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_random = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_fixed = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_fixed = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_fixed = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_hr = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_hr = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_hr = []  # for each specified max throughput in 'max_network_throughput_list'
    df_split_idx_list_hr = [] # for each specified max throughput in 'max_network_throughput_list'

    for max_throughput in max_network_throughput_list:
        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, _ = parse_kpis('rl/ddqn', total_episodes_train['rl/ddqn'],
                                                                             max_throughput)
        df_inference_time_list_ddqn.append(df_inference_time)
        df_ue_energy_comp_list_ddqn.append(df_ue_energy_comp)
        df_ue_energy_comm_list_ddqn.append(df_ue_energy_comm)
        print('ddqn done')

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, _ = parse_kpis('rl/a2c', total_episodes_train['rl/a2c'],
                                                                             max_throughput)
        df_inference_time_list_a2c.append(df_inference_time)
        df_ue_energy_comp_list_a2c.append(df_ue_energy_comp)
        df_ue_energy_comm_list_a2c.append(df_ue_energy_comm)
        print('a2c done')

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, _ = parse_kpis('rl/ppo',
                                                                                   total_episodes_train['rl/ppo'],
                                                                                   max_throughput)
        df_inference_time_list_ppo.append(df_inference_time)
        df_ue_energy_comp_list_ppo.append(df_ue_energy_comp)
        df_ue_energy_comm_list_ppo.append(df_ue_energy_comm)
        print('ppo done')

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, _ = parse_kpis('optimum', 9, max_throughput)
        df_inference_time_list_opt.append(df_inference_time)
        df_ue_energy_comp_list_opt.append(df_ue_energy_comp)
        df_ue_energy_comm_list_opt.append(df_ue_energy_comm)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, _ = parse_kpis('random', 300, max_throughput)
        df_inference_time_list_random.append(df_inference_time)
        df_ue_energy_comp_list_random.append(df_ue_energy_comp)
        df_ue_energy_comm_list_random.append(df_ue_energy_comm)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, _ = parse_kpis('fixed', 9, max_throughput)
        df_inference_time_list_fixed.append(df_inference_time)
        df_ue_energy_comp_list_fixed.append(df_ue_energy_comp)
        df_ue_energy_comm_list_fixed.append(df_ue_energy_comm)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, df_split_idx = parse_kpis('heuristic', 300, max_throughput)
        df_inference_time_list_hr.append(df_inference_time)
        df_ue_energy_comp_list_hr.append(df_ue_energy_comp)
        df_ue_energy_comm_list_hr.append(df_ue_energy_comm)
        df_split_idx_list_hr.append(df_split_idx)

    df_all_inference_time = [df_inference_time_list_ddqn, df_inference_time_list_a2c, df_inference_time_list_ppo,
                             df_inference_time_list_opt,
                             df_inference_time_list_random, df_inference_time_list_fixed, df_inference_time_list_hr]
    df_all_ue_energy_comp = [df_ue_energy_comp_list_ddqn, df_ue_energy_comp_list_a2c, df_ue_energy_comp_list_ppo,
                             df_ue_energy_comp_list_opt,
                             df_ue_energy_comp_list_random, df_ue_energy_comp_list_fixed, df_ue_energy_comp_list_hr]
    df_all_ue_energy_comm = [df_ue_energy_comm_list_ddqn, df_ue_energy_comm_list_a2c, df_ue_energy_comm_list_ppo,
                             df_ue_energy_comm_list_opt,
                             df_ue_energy_comm_list_random, df_ue_energy_comm_list_fixed, df_ue_energy_comm_list_hr]
    plot_kpis_vs_max_throughput(df_all_inference_time, df_all_ue_energy_comp, df_all_ue_energy_comm,
                                    max_network_throughput_list, n_episodes_to_train, total_episodes_train, df_split_idx_list_hr)

if __name__ == '__main__':
    main()