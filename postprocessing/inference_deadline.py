import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from postprocessing.network_throughput import return_ci_95, return_ci_90, _plot_errorbar_series
from postprocessing.plot_system_kpis import return_order, read_kpis_from_files, parse_episode_number

# Evenly-spaced x-offsets (in seconds) for each series, symmetric around 0 - same linspace-based scheme
# as network_throughput.py's _SERIES_OFFSETS, but sized for THIS axis: inference_deadline_list points are
# spaced 0.025 apart (0.225, 0.25, 0.275, ...), vs. network_throughput's 50 MB/s spacing, so the offsets
# here are proportionally much smaller. The six hand-picked offsets this replaces (-0.006/-0.003/0/+0.001/
# +0.003/+0.006) had the same uneven-spacing issue network_throughput.py's did before that fix.
_SERIES_ORDER = ['opt', 'ddqn', 'a2c', 'ppo', 'heuristic', 'random', 'fixed']
_JITTER_SPREAD = 0.0075  # +/- seconds from center; total spread stays well under the 0.025 point spacing
_SERIES_OFFSETS = dict(zip(_SERIES_ORDER, np.linspace(-_JITTER_SPREAD, _JITTER_SPREAD, len(_SERIES_ORDER))))


def parse_kpis(folder, n_episodes, inference_deadline):
    """
    Function to read and parse kpis into a 2D pandas DataFrame.
    Args:
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        n_episodes (int): number of episodes per algorithm to print
        inference_deadline (float): the inference deadline setting of the desired kpi

    Returns:
        Tuple: (dataframes containing inference time, ue computation and communication energy and energy credit)
    """
    if folder == 'rl/ddqn':
        order = return_order(n_episodes=1000)
    elif folder == 'random':
        order = return_order(n_episodes=100)
    else:
        order = return_order(n_episodes)
    inference_times_all_episodes = []
    ue_energy_comp_all_episodes = []
    ue_energy_comm_all_episodes = []
    energy_credit_all_episodes = []
    y_net_all_episodes = []
    top1_all_episodes = []
    flops_off_all_episodes = []
    time_steps = []

    #if inference_deadline == 0.25 and folder == 'random':
    #    n_episodes = 200

    for episode in range(1, n_episodes + 1):
        episode_count = parse_episode_number(order, episode)
        kpi_type = 'inference_time'
        time_steps, inference_times_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
        inference_times_all_episodes.append(inference_times_per_episode)
        kpi_type = 'ue_energy_comp'
        time_steps, ue_energy_comp_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
        ue_energy_comp_all_episodes.append(ue_energy_comp_per_episode)
        kpi_type = 'ue_energy_comm'
        time_steps, ue_energy_comm_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
        ue_energy_comm_all_episodes.append(ue_energy_comm_per_episode)
        kpi_type = 'energy_credit'
        time_steps, energy_credit_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
        energy_credit_all_episodes.append(energy_credit_per_episode)
        kpi_type = 'top1'
        time_steps, top1_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
        top1_all_episodes.append(top1_per_episode)
        if folder == 'rl/ddqn':
            kpi_type = 'y_net'
            time_steps, y_net_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
            y_net_all_episodes.append(y_net_per_episode)
            kpi_type = 'flops_off'
            #time_steps, flops_off_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
            #flops_off_all_episodes.append(flops_off_per_episode)

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
    df_top1 = pd.DataFrame(top1_all_episodes, columns=time_steps,
                                    index=[ep for ep in range(1, n_episodes + 1)])
    #df_flops_off = pd.DataFrame(flops_off_all_episodes, columns=time_steps,
    #                                index=[ep for ep in range(1, n_episodes + 1)])

    return df_inference_time, df_ue_energy_comp, df_ue_energy_comm, df_energy_credit, df_y_net, df_top1


def plot_kpis_vs_inference_deadline(df_all_inference_time, df_all_ue_energy_comp, df_all_ue_energy_comm,
                                    inference_deadline_list, n_episodes_to_train, total_episodes_train, df_all_top1):
    """
    Script to plot the comparison between inference time vs energy credit usage, or ue energy vs energy credit usage.
    Args:
        df_all_inference_time (list of pandas DataFrame): list of 2D dataframes containing of inference time logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_all_ue_energy_comp (list of pandas DataFrame): list of 2D dataframes containing of ue_energy_comp logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_all_ue_energy_comm (list of pandas DataFrame): list of 2D dataframes containing of ue_energy_comm logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_all_top1 (list of pandas DataFrame): list of 2D dataframes containing of top1 logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        inference_deadline_list (list): list of inference deadline values to plot.

    Returns:

    """
    #r = np.arange(len(algorithms))  # the label locations
    #width = 0.25  # the width of the bars
    #multiplier = 0
    df_inference_time_ddqn_list = df_all_inference_time[0]   # ddqn
    df_ue_energy_comp_ddqn_list = df_all_ue_energy_comp[0]   # ddqn
    df_ue_energy_comm_ddqn_list = df_all_ue_energy_comm[0]   # ddqn
    df_top1_ddqn_list = df_all_top1[0]                       # ddqn
    df_inference_time_a2c_list = df_all_inference_time[1]   # a2c
    df_ue_energy_comp_a2c_list = df_all_ue_energy_comp[1]   # a2c
    df_ue_energy_comm_a2c_list = df_all_ue_energy_comm[1]   # a2c
    df_top1_a2c_list = df_all_top1[1]                       # a2c
    df_inference_time_ppo_list = df_all_inference_time[2]  # ppo
    df_ue_energy_comp_ppo_list = df_all_ue_energy_comp[2]  # ppo
    df_ue_energy_comm_ppo_list = df_all_ue_energy_comm[2]  # ppo
    df_top1_ppo_list = df_all_top1[2]  # ppo
    df_inference_time_opt_list = df_all_inference_time[3]   # opt
    df_ue_energy_comp_opt_list = df_all_ue_energy_comp[3]   # opt
    df_ue_energy_comm_opt_list = df_all_ue_energy_comm[3]   # opt
    df_top1_opt_list = df_all_top1[3]                       # opt
    df_inference_time_random_list = df_all_inference_time[4]    # random
    df_ue_energy_comp_random_list = df_all_ue_energy_comp[4]    # random
    df_ue_energy_comm_random_list = df_all_ue_energy_comm[4]    # random
    df_top1_random_list = df_all_top1[4]                        # random
    df_inference_time_fixed_list = df_all_inference_time[5]     # fixed
    df_ue_energy_comp_fixed_list = df_all_ue_energy_comp[5]     # fixed
    df_ue_energy_comm_fixed_list = df_all_ue_energy_comm[5]     # fixed
    df_top1_fixed_list = df_all_top1[5]                         # fixed
    df_inference_time_hr_list = df_all_inference_time[6]  # heuristic
    df_ue_energy_comp_hr_list = df_all_ue_energy_comp[6]  # heuristic
    df_ue_energy_comm_hr_list = df_all_ue_energy_comm[6]  # heuristic
    df_top1_hr_list = df_all_top1[6]  # heuristic

    n_episodes_aft_train_ddqn = total_episodes_train['rl/ddqn'] - n_episodes_to_train['rl/ddqn']
    n_episodes_aft_train_a2c = total_episodes_train['rl/a2c'] - n_episodes_to_train['rl/a2c']
    n_episodes_aft_train_ppo = total_episodes_train['rl/ppo'] - n_episodes_to_train['rl/ppo']
    fig, ax = plt.subplots()
    inference_time_mean_per_deadline_ddqn = []
    ue_energy_comp_mean_per_deadline_ddqn = []
    ue_energy_comp_sd_per_deadline_ddqn = []
    ue_energy_comp_ci_per_deadline_ddqn = []
    ue_energy_comm_mean_per_deadline_ddqn = []
    top1_mean_per_deadline_ddqn = []

    inference_time_mean_per_deadline_a2c = []
    ue_energy_comp_mean_per_deadline_a2c = []
    ue_energy_comp_sd_per_deadline_a2c = []
    ue_energy_comp_ci_per_deadline_a2c = []
    ue_energy_comm_mean_per_deadline_a2c = []
    top1_mean_per_deadline_a2c = []

    inference_time_mean_per_deadline_ppo = []
    ue_energy_comp_mean_per_deadline_ppo = []
    ue_energy_comp_sd_per_deadline_ppo = []
    ue_energy_comp_ci_per_deadline_ppo = []
    ue_energy_comm_mean_per_deadline_ppo = []
    top1_mean_per_deadline_ppo = []

    inference_time_mean_per_deadline_opt = []
    ue_energy_comp_mean_per_deadline_opt = []
    ue_energy_comp_sd_per_deadline_opt = []
    ue_energy_comp_ci_per_deadline_opt = []
    ue_energy_comm_mean_per_deadline_opt = []
    top1_mean_per_deadline_opt = []

    inference_time_mean_per_deadline_random = []
    ue_energy_comp_mean_per_deadline_random = []
    ue_energy_comp_sd_per_deadline_random = []
    ue_energy_comp_ci_per_deadline_random = []
    ue_energy_comm_mean_per_deadline_random = []
    top1_mean_per_deadline_random = []

    inference_time_mean_per_deadline_fixed = []
    ue_energy_comp_mean_per_deadline_fixed = []
    ue_energy_comp_sd_per_deadline_fixed = []
    ue_energy_comp_ci_per_deadline_fixed = []
    ue_energy_comm_mean_per_deadline_fixed = []
    top1_mean_per_deadline_fixed = []

    inference_time_mean_per_deadline_hr = []
    ue_energy_comp_mean_per_deadline_hr = []
    ue_energy_comp_sd_per_deadline_hr = []
    ue_energy_comp_ci_per_deadline_hr = []
    ue_energy_comm_mean_per_deadline_hr = []
    top1_mean_per_deadline_hr = []
    # extract data and store means
    for i, deadline in enumerate(inference_deadline_list):
        df_inference_time_ddqn = df_inference_time_ddqn_list[i]
        df_ue_energy_comp_ddqn = df_ue_energy_comp_ddqn_list[i]
        df_ue_energy_comm_ddqn = df_ue_energy_comm_ddqn_list[i]
        df_top1_ddqn = df_top1_ddqn_list[i]
        # then calculate the means
        df_inference_time_ddqn['mean'] = df_inference_time_ddqn.mean(axis=1)
        inference_time_mean_per_deadline_ddqn.append(df_inference_time_ddqn['mean'].iloc[n_episodes_aft_train_ddqn:total_episodes_train['rl/ddqn']].mean())
        df_ue_energy_comp_ddqn['mean'] = df_ue_energy_comp_ddqn.mean(axis=1)
        ue_energy_comp_mean_per_deadline_ddqn.append(df_ue_energy_comp_ddqn['mean'].iloc[n_episodes_aft_train_ddqn:total_episodes_train['rl/ddqn']].mean())
        ue_energy_comp_ddqn_sd = df_ue_energy_comp_ddqn['mean'].std()
        ue_energy_comp_ddqn_ci = return_ci_90(ue_energy_comp_ddqn_sd)
        ue_energy_comp_sd_per_deadline_ddqn.append(ue_energy_comp_ddqn_sd)
        ue_energy_comp_ci_per_deadline_ddqn.append(ue_energy_comp_ddqn_ci)
        df_ue_energy_comm_ddqn['mean'] = df_ue_energy_comm_ddqn.mean(axis=1)
        ue_energy_comm_mean_per_deadline_ddqn.append(df_ue_energy_comm_ddqn['mean'].iloc[n_episodes_aft_train_ddqn:total_episodes_train['rl/ddqn']].mean())
        #sum_ue_energy_per_deadline.append(df_ue_energy_comp_ddqn['mean'][n_episodes_bef_train:].mean() +
        #                                  df_ue_energy_comm_ddqn['mean'][n_episodes_bef_train:])
        df_top1_ddqn['mean'] = df_top1_ddqn.mean(axis=1)
        top1_mean_per_deadline_ddqn.append(df_top1_ddqn['mean'].iloc[n_episodes_aft_train_ddqn:total_episodes_train['rl/ddqn']].mean())

        # then a2c
        df_inference_time_a2c = df_inference_time_a2c_list[i]
        df_ue_energy_comp_a2c = df_ue_energy_comp_a2c_list[i]
        df_ue_energy_comm_a2c = df_ue_energy_comm_a2c_list[i]
        df_top1_a2c = df_top1_a2c_list[i]
        df_inference_time_a2c['mean'] = df_inference_time_a2c.mean(axis=1)
        inference_time_mean_per_deadline_a2c.append(
            df_inference_time_a2c['mean'].iloc[n_episodes_aft_train_a2c:total_episodes_train['rl/a2c']].mean())
        df_ue_energy_comp_a2c['mean'] = df_ue_energy_comp_a2c.mean(axis=1)
        ue_energy_comp_mean_per_deadline_a2c.append(
            df_ue_energy_comp_a2c['mean'].iloc[n_episodes_aft_train_a2c:total_episodes_train['rl/a2c']].mean())
        ue_energy_comp_a2c_sd = df_ue_energy_comp_a2c['mean'].std()
        ue_energy_comp_a2c_ci = return_ci_90(ue_energy_comp_a2c_sd)
        ue_energy_comp_sd_per_deadline_a2c.append(ue_energy_comp_a2c_sd)
        ue_energy_comp_ci_per_deadline_a2c.append(ue_energy_comp_a2c_ci)
        df_ue_energy_comm_a2c['mean'] = df_ue_energy_comm_a2c.mean(axis=1)
        ue_energy_comm_mean_per_deadline_a2c.append(
            df_ue_energy_comm_a2c['mean'].iloc[n_episodes_aft_train_a2c:total_episodes_train['rl/a2c']].mean())
        df_top1_a2c['mean'] = df_top1_a2c.mean(axis=1)
        top1_mean_per_deadline_a2c.append(df_top1_a2c['mean'].iloc[n_episodes_aft_train_a2c:total_episodes_train['rl/a2c']].mean())

        # then ppo
        df_inference_time_ppo = df_inference_time_ppo_list[i]
        df_ue_energy_comp_ppo = df_ue_energy_comp_ppo_list[i]
        df_ue_energy_comm_ppo = df_ue_energy_comm_ppo_list[i]
        df_top1_ppo = df_top1_ppo_list[i]
        df_inference_time_ppo['mean'] = df_inference_time_ppo.mean(axis=1)
        inference_time_mean_per_deadline_ppo.append(
            df_inference_time_ppo['mean'].iloc[n_episodes_aft_train_ppo:total_episodes_train['rl/ppo']].mean())
        df_ue_energy_comp_ppo['mean'] = df_ue_energy_comp_ppo.mean(axis=1)
        ue_energy_comp_mean_per_deadline_ppo.append(
            df_ue_energy_comp_ppo['mean'].iloc[n_episodes_aft_train_ppo:total_episodes_train['rl/ppo']].mean())
        ue_energy_comp_ppo_sd = df_ue_energy_comp_ppo['mean'].std()
        ue_energy_comp_ppo_ci = return_ci_90(ue_energy_comp_ppo_sd)
        ue_energy_comp_sd_per_deadline_ppo.append(ue_energy_comp_ppo_sd)
        ue_energy_comp_ci_per_deadline_ppo.append(ue_energy_comp_ppo_ci)
        df_ue_energy_comm_ppo['mean'] = df_ue_energy_comm_ppo.mean(axis=1)
        ue_energy_comm_mean_per_deadline_ppo.append(
            df_ue_energy_comm_ppo['mean'].iloc[n_episodes_aft_train_ppo:total_episodes_train['rl/ppo']].mean())
        df_top1_ppo['mean'] = df_top1_ppo.mean(axis=1)
        top1_mean_per_deadline_ppo.append(
            df_top1_ppo['mean'].iloc[n_episodes_aft_train_ppo:total_episodes_train['rl/ppo']].mean())

        # then optimum
        df_inference_time_opt = df_inference_time_opt_list[i]
        df_ue_energy_comp_opt = df_ue_energy_comp_opt_list[i]
        df_ue_energy_comm_opt = df_ue_energy_comm_opt_list[i]
        df_top1_opt = df_top1_opt_list[i]
        df_inference_time_opt['mean'] = df_inference_time_opt.mean(axis=1)
        inference_time_mean_per_deadline_opt.append(df_inference_time_opt['mean'].mean())
        df_ue_energy_comp_opt['mean'] = df_ue_energy_comp_opt.mean(axis=1)
        ue_energy_comp_mean_per_deadline_opt.append(df_ue_energy_comp_opt['mean'].mean())
        ue_energy_comp_opt_sd = df_ue_energy_comp_opt['mean'].std()
        ue_energy_comp_opt_ci = return_ci_90(ue_energy_comp_opt_sd)
        ue_energy_comp_sd_per_deadline_opt.append(ue_energy_comp_opt_sd)
        ue_energy_comp_ci_per_deadline_opt.append(ue_energy_comp_opt_ci)
        df_ue_energy_comm_opt['mean'] = df_ue_energy_comm_opt.mean(axis=1)
        ue_energy_comm_mean_per_deadline_opt.append(df_ue_energy_comm_opt['mean'].mean())
        df_top1_opt['mean'] = df_top1_opt.mean(axis=1)
        top1_mean_per_deadline_opt.append(df_top1_opt['mean'].mean())

        # then random
        df_inference_time_random = df_inference_time_random_list[i]
        df_ue_energy_comp_random = df_ue_energy_comp_random_list[i]
        df_ue_energy_comm_random = df_ue_energy_comm_random_list[i]
        df_top1_random = df_top1_random_list[i]
        df_inference_time_random['mean'] = df_inference_time_random.mean(axis=1)
        inference_time_mean_per_deadline_random.append(df_inference_time_random['mean'].mean())
        df_ue_energy_comp_random['mean'] = df_ue_energy_comp_random.mean(axis=1)
        ue_energy_comp_mean_per_deadline_random.append(df_ue_energy_comp_random['mean'].mean())
        ue_energy_comp_random_sd = df_ue_energy_comp_random['mean'].std()
        ue_energy_comp_random_ci = return_ci_90(ue_energy_comp_random_sd)
        ue_energy_comp_sd_per_deadline_random.append(ue_energy_comp_random_sd)
        ue_energy_comp_ci_per_deadline_random.append(ue_energy_comp_random_ci)
        df_ue_energy_comm_random['mean'] = df_ue_energy_comm_random.mean(axis=1)
        ue_energy_comm_mean_per_deadline_random.append(df_ue_energy_comm_random['mean'].mean())
        df_top1_random['mean'] = df_top1_random.mean(axis=1)
        top1_mean_per_deadline_random.append(df_top1_random['mean'].mean())

        # # then fixed
        df_inference_time_fixed = df_inference_time_fixed_list[i]
        df_ue_energy_comp_fixed = df_ue_energy_comp_fixed_list[i]
        df_ue_energy_comm_fixed = df_ue_energy_comm_fixed_list[i]
        df_top1_fixed = df_top1_fixed_list[i]
        df_inference_time_fixed['mean'] = df_inference_time_fixed.mean(axis=1)
        inference_time_mean_per_deadline_fixed.append(df_inference_time_fixed['mean'].mean())
        df_ue_energy_comp_fixed['mean'] = df_ue_energy_comp_fixed.mean(axis=1)
        ue_energy_comp_mean_per_deadline_fixed.append(df_ue_energy_comp_fixed['mean'].mean())
        ue_energy_comp_fixed_sd = df_ue_energy_comp_fixed['mean'].std()
        ue_energy_comp_fixed_ci = return_ci_90(ue_energy_comp_fixed_sd)
        ue_energy_comp_sd_per_deadline_fixed.append(ue_energy_comp_fixed_sd)
        ue_energy_comp_ci_per_deadline_fixed.append(ue_energy_comp_fixed_ci)
        df_ue_energy_comm_fixed['mean'] = df_ue_energy_comm_fixed.mean(axis=1)
        ue_energy_comm_mean_per_deadline_fixed.append(df_ue_energy_comm_fixed['mean'].mean())
        df_top1_fixed['mean'] = df_top1_fixed.mean(axis=1)
        top1_mean_per_deadline_fixed.append(df_top1_fixed['mean'].mean())

        # finally heuristic
        df_inference_time_hr = df_inference_time_hr_list[i]
        df_ue_energy_comp_hr = df_ue_energy_comp_hr_list[i]
        df_ue_energy_comm_hr = df_ue_energy_comm_hr_list[i]
        df_top1_hr = df_top1_hr_list[i]
        df_inference_time_hr['mean'] = df_inference_time_hr.mean(axis=1)
        inference_time_mean_per_deadline_hr.append(df_inference_time_hr['mean'].mean())
        df_ue_energy_comp_hr['mean'] = df_ue_energy_comp_hr.mean(axis=1)
        ue_energy_comp_mean_per_deadline_hr.append(df_ue_energy_comp_hr['mean'].mean())
        ue_energy_comp_hr_sd = df_ue_energy_comp_hr['mean'].std()
        ue_energy_comp_hr_ci = return_ci_90(ue_energy_comp_hr_sd)
        ue_energy_comp_sd_per_deadline_hr.append(ue_energy_comp_hr_sd)
        ue_energy_comp_ci_per_deadline_hr.append(ue_energy_comp_hr_ci)
        df_ue_energy_comm_hr['mean'] = df_ue_energy_comm_hr.mean(axis=1)
        ue_energy_comm_mean_per_deadline_hr.append(df_ue_energy_comm_hr['mean'].mean())
        df_top1_hr['mean'] = df_top1_hr.mean(axis=1)
        top1_mean_per_deadline_hr.append(df_top1_hr['mean'].mean())
    # sum_ue_energy_per_deadline_ddqn = np.add(ue_energy_comp_mean_per_deadline_ddqn, ue_energy_comm_mean_per_deadline_ddqn)
    # sum_ue_energy_per_deadline_opt = np.add(ue_energy_comp_mean_per_deadline_opt, ue_energy_comm_mean_per_deadline_opt)
    # sum_ue_energy_per_deadline_random = np.add(ue_energy_comp_mean_per_deadline_random, ue_energy_comm_mean_per_deadline_random)
    # sum_ue_energy_per_deadline_fixed = np.add(ue_energy_comp_mean_per_deadline_fixed, ue_energy_comm_mean_per_deadline_fixed)
    #print(sum_ue_energy_per_deadline_ddqn)
    #print(sum_ue_energy_per_deadline_opt)

    # sum energy
    #ax.plot(inference_deadline_list, sum_ue_energy_per_deadline_opt, color='#072140', marker='o', label='opt')
    #ax.plot(inference_deadline_list, sum_ue_energy_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    #ax.plot(inference_deadline_list, sum_ue_energy_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    #ax.plot(inference_deadline_list, sum_ue_energy_per_deadline_fixed, color='#8F81EA', marker='o', label='fixed')

    #print(sum_ue_energy_per_deadline_ddqn)
    # comp energy
    # ax.plot(inference_deadline_list, ue_energy_comp_mean_per_deadline_opt, color='#072140', marker='o', label='opt')
    # ax.plot(inference_deadline_list, ue_energy_comp_mean_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    # ax.plot(inference_deadline_list, ue_energy_comp_mean_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    # ax.plot(inference_deadline_list, ue_energy_comp_mean_per_deadline_fixed, color='#8F81EA', marker='o', label='fixed')
    # ax1 = ax.twinx()
    # ax1.plot(inference_deadline_list, top1_mean_per_deadline_opt, color='#072140', marker='D', label='opt')
    # ax1.plot(inference_deadline_list, top1_mean_per_deadline_ddqn, color='#165DB1', marker='D', label='ddqn')
    # ax1.plot(inference_deadline_list, top1_mean_per_deadline_random, color='#9ABCE4', marker='D', label='random')
    # ax1.plot(inference_deadline_list, top1_mean_per_deadline_fixed, color='#8F81EA', marker='D', label='fixed')

    #ax1.set_ylabel('Top 1 accuracy confidence (%)')
    #print(ue_energy_comp_mean_per_deadline_random)
    #print(ue_energy_comp_mean_per_deadline_ddqn)
    # comm energy
    #ax.plot(inference_deadline_list, ue_energy_comm_mean_per_deadline_opt, color='#072140', marker='o', label='opt')
    #ax.plot(inference_deadline_list, ue_energy_comm_mean_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    #ax.plot(inference_deadline_list, ue_energy_comm_mean_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    #ax.plot(inference_deadline_list, ue_energy_comm_mean_per_deadline_fixed, color='#8F81EA', marker='o', label='fixed')
    # top 1 accuracy confidence
    #ax.plot(inference_deadline_list, top1_mean_per_deadline_opt, color='#072140', marker='o', label='opt')
    #ax.plot(inference_deadline_list, top1_mean_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    #ax.plot(inference_deadline_list, top1_mean_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    # inference time
    # ax.plot(inference_deadline_list, inference_time_mean_per_deadline_opt, color='#072140', marker='o', label='opt')
    # ax.plot(inference_deadline_list, inference_time_mean_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    # ax.plot(inference_deadline_list, inference_time_mean_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    _plot_errorbar_series(inference_deadline_list, ue_energy_comp_mean_per_deadline_opt,
                           ue_energy_comp_ci_per_deadline_opt, "#072140", 'opt', _SERIES_OFFSETS['opt'])
    _plot_errorbar_series(inference_deadline_list, ue_energy_comp_mean_per_deadline_ddqn,
                           ue_energy_comp_ci_per_deadline_ddqn, "#114584", 'ddqn', _SERIES_OFFSETS['ddqn'])
    _plot_errorbar_series(inference_deadline_list, ue_energy_comp_mean_per_deadline_a2c,
                           ue_energy_comp_ci_per_deadline_a2c, "#165DB1", 'a2c', _SERIES_OFFSETS['a2c'])
    _plot_errorbar_series(inference_deadline_list, ue_energy_comp_mean_per_deadline_ppo,
                           ue_energy_comp_ci_per_deadline_ppo, "#475058", 'ppo', _SERIES_OFFSETS['ppo'])
    _plot_errorbar_series(inference_deadline_list, ue_energy_comp_mean_per_deadline_hr,
                           ue_energy_comp_ci_per_deadline_hr, "#9B468D", 'heuristic', _SERIES_OFFSETS['heuristic'])
    _plot_errorbar_series(inference_deadline_list, ue_energy_comp_mean_per_deadline_random,
                           ue_energy_comp_ci_per_deadline_random, "#9ABCE4", 'random', _SERIES_OFFSETS['random'])
    _plot_errorbar_series(inference_deadline_list, ue_energy_comp_mean_per_deadline_fixed,
                           ue_energy_comp_ci_per_deadline_fixed, "#8F81EA", 'fixed', _SERIES_OFFSETS['fixed'])
    print(ue_energy_comp_mean_per_deadline_hr)
    print(ue_energy_comp_mean_per_deadline_fixed)
    ax.set_xlabel('Inference deadline (s)')
    ax.set_ylabel('UE energy comp (J)')
    #ax.set_ylabel('Inference time (s)')
    #ax.set_ylabel('Top 1 accuracy confidence (%)')
    plt.grid()
    plt.legend()
    plt.savefig('results/journal/comp_energy_vs_deadlines_all.png')
    plt.savefig('results/journal/comp_energy_vs_deadlines_all.svg')
    #plt.show()

def main():
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    inference_deadline_list = [0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]
    # specifies the episodes of convergence of ddqn
    n_episodes_to_train = {'rl/ddqn': 4000, 'rl/a2c': 3000, 'rl/ppo': 2500}
    total_episodes_train = {'rl/ddqn': 5000, 'rl/a2c': 5000, 'rl/ppo': 5000}
    df_inference_time_list_ddqn = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comp_list_ddqn = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comm_list_ddqn = []  # for each specified deadline in 'inference_deadline_list'
    df_top1_list_ddqn = []            # for each specified deadline in 'inference_deadline_list'
    df_inference_time_list_a2c = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comp_list_a2c = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comm_list_a2c = []  # for each specified deadline in 'inference_deadline_list'
    df_top1_list_a2c = []            # for each specified deadline in 'inference_deadline_list'
    df_inference_time_list_ppo = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comp_list_ppo = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comm_list_ppo = []  # for each specified deadline in 'inference_deadline_list'
    df_top1_list_ppo = []  # for each specified deadline in 'inference_deadline_list'
    df_inference_time_list_hr = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comp_list_hr = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comm_list_hr = []  # for each specified deadline in 'inference_deadline_list'
    df_top1_list_hr = []  # for each specified deadline in 'inference_deadline_list'
    df_inference_time_list_opt = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comp_list_opt = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comm_list_opt = []  # for each specified deadline in 'inference_deadline_list'
    df_top1_list_opt = []           # for each specified deadline in 'inference_deadline_list'
    df_inference_time_list_random = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comp_list_random = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comm_list_random = []  # for each specified deadline in 'inference_deadline_list'
    df_top1_list_random = []            # for each specified deadline in 'inference_deadline_list'
    df_inference_time_list_fixed = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comp_list_fixed = []  # for each specified deadline in 'inference_deadline_list'
    df_ue_energy_comm_list_fixed = []  # for each specified deadline in 'inference_deadline_list'
    df_top1_list_fixed = []             # for each specified deadline in 'inference_deadline_list'

    for deadline in inference_deadline_list:
        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, df_top1 = parse_kpis('rl/ddqn', total_episodes_train['rl/ddqn'],
                                                                             deadline)
        df_inference_time_list_ddqn.append(df_inference_time)
        df_ue_energy_comp_list_ddqn.append(df_ue_energy_comp)
        df_ue_energy_comm_list_ddqn.append(df_ue_energy_comm)
        df_top1_list_ddqn.append(df_top1)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, df_top1 = parse_kpis('rl/a2c', total_episodes_train['rl/a2c'],
                                                                             deadline)
        df_inference_time_list_a2c.append(df_inference_time)
        df_ue_energy_comp_list_a2c.append(df_ue_energy_comp)
        df_ue_energy_comm_list_a2c.append(df_ue_energy_comm)
        df_top1_list_a2c.append(df_top1)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, df_top1 = parse_kpis('rl/ppo', total_episodes_train['rl/ppo'],
                                                                             deadline)
        df_inference_time_list_ppo.append(df_inference_time)
        df_ue_energy_comp_list_ppo.append(df_ue_energy_comp)
        df_ue_energy_comm_list_ppo.append(df_ue_energy_comm)
        df_top1_list_ppo.append(df_top1)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, df_top1 = parse_kpis('heuristic', 300,
                                                                                             deadline)
        df_inference_time_list_hr.append(df_inference_time)
        df_ue_energy_comp_list_hr.append(df_ue_energy_comp)
        df_ue_energy_comm_list_hr.append(df_ue_energy_comm)
        df_top1_list_hr.append(df_top1)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, df_top1 = parse_kpis('optimum', 9, deadline)
        df_inference_time_list_opt.append(df_inference_time)
        df_ue_energy_comp_list_opt.append(df_ue_energy_comp)
        df_ue_energy_comm_list_opt.append(df_ue_energy_comm)
        df_top1_list_opt.append(df_top1)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, df_top1 = parse_kpis('random', 300, deadline)
        df_inference_time_list_random.append(df_inference_time)
        df_ue_energy_comp_list_random.append(df_ue_energy_comp)
        df_ue_energy_comm_list_random.append(df_ue_energy_comm)
        df_top1_list_random.append(df_top1)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _, df_top1 = parse_kpis('fixed', 9, deadline)
        df_inference_time_list_fixed.append(df_inference_time)
        df_ue_energy_comp_list_fixed.append(df_ue_energy_comp)
        df_ue_energy_comm_list_fixed.append(df_ue_energy_comm)
        df_top1_list_fixed.append(df_top1)

    df_all_inference_time = [df_inference_time_list_ddqn, df_inference_time_list_a2c, df_inference_time_list_ppo,
                             df_inference_time_list_opt,
                             df_inference_time_list_random, df_inference_time_list_fixed, df_inference_time_list_hr]
    df_all_ue_energy_comp = [df_ue_energy_comp_list_ddqn, df_ue_energy_comp_list_a2c, df_ue_energy_comp_list_ppo,
                             df_ue_energy_comp_list_opt,
                             df_ue_energy_comp_list_random, df_ue_energy_comp_list_fixed, df_ue_energy_comp_list_hr]
    df_all_ue_energy_comm = [df_ue_energy_comm_list_ddqn, df_ue_energy_comm_list_a2c, df_ue_energy_comm_list_ppo,
                             df_ue_energy_comm_list_opt,
                             df_ue_energy_comm_list_random, df_ue_energy_comm_list_fixed, df_ue_energy_comm_list_hr]
    df_all_top1 = [df_top1_list_ddqn, df_top1_list_a2c, df_top1_list_ppo, df_top1_list_opt, df_top1_list_random, df_top1_list_fixed,
                   df_top1_list_hr]
    plot_kpis_vs_inference_deadline(df_all_inference_time, df_all_ue_energy_comp, df_all_ue_energy_comm,
                                    inference_deadline_list, n_episodes_to_train, total_episodes_train, df_all_top1)

if __name__ == '__main__':
    main()