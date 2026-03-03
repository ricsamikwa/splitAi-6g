import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.logging_utils import return_order, parse_episode_number


def read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline):
    """
    Function to read system KPIs (inference time, ue computation and communication energy) for each episode.

    Args:
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        kpi_type (str): The kpi to read e.g. inference_time
        episode_count (int): The episode number
        inference_deadline (float or NoneType): the inference deadline of the desired kpi
    Returns:
        Tuple: (time step when kpi was recorded as list, kpi as list )
    """
    if inference_deadline is None:
        file = 'logs/{}/system/{}_{}.csv'.format(folder, kpi_type, episode_count)
    else:
        file = 'logs/{}/comparison/lat_{}/system/{}_{}.csv'.format(folder, inference_deadline, kpi_type, episode_count)
    data_timestep = []
    data_kpi = []
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for k, item in enumerate(reader):
            if k != 0:
                data_timestep.append(int(item[0]))
                data_kpi.append(float(item[1]))
    return data_timestep, data_kpi

def get_confidence_interval(sd, size):
    """
    Compute 95% confidence interval

    Args:
        sd (float): standard deviation
        size (int): sample size

    Returns:
        95% confidence interval of a group of samples
    """
    return 1.960 * sd / size



def plot_kpis_tradeoff_optimum(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list, omega_list):
    """
    Script to plot the tradeoff curve between inference time and ue energy for the optimum case for different values of
    the weight, omega. This tradeoff could be analysed using static values of the system parameters.
    Args:
        df_inference_time_list (list of pandas DataFrame): list of 2D dataframes containing of inference time logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_ue_energy_comp_list (list of pandas DataFrame): list of 2D dataframes containing of ue_energy_comp logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_ue_energy_comm_list (list of pandas DataFrame): list of 2D dataframes containing of ue_energy_comm logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        omega_list (list): list of values of omega to plot.

    Returns:

    """

    inference_time_mean_per_omega = []
    ue_energy_mean_per_omega = []
    for i, omega in enumerate(omega_list):
        # extract dataframes for the current value of omega
        df_inference_time = df_inference_time_list[i]
        df_ue_energy_comp = df_ue_energy_comp_list[i]
        df_ue_energy_comm = df_ue_energy_comm_list[i]
        #print(df_inference_time.mean(axis=1))
        #print(df_ue_energy_comp)
        #print(df_ue_energy_comm)

        # store the mean of the episodes for all kpis
        inference_time_mean_per_omega.append(omega * df_inference_time.mean(axis=1))
        sum_ue_energies = df_ue_energy_comp + df_ue_energy_comm
        ue_energy_mean_per_omega.append((1 - omega) * sum_ue_energies.mean(axis=1))

    fig, ax1 = plt.subplots()
    #cmap = plt.get_cmap('viridis')
    #colors = cmap(np.linspace(0, 0.7, n_episodes_to_plot))
    color = 'tab:blue'
    ax1.set_xlabel('omega')
    ax1.set_ylabel('Weighted inference latency in s')
    # plot on one axis first
    #print(inference_time_mean_per_omega)
    ax1.plot(omega_list, inference_time_mean_per_omega, color=color, marker='o', label='Inference latency')
    # instantiate a second Axes that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Weighted UE energy in J')
    #print(ue_energy_mean_per_omega)
    ax2.plot(omega_list, ue_energy_mean_per_omega, color=color, marker='D', label='UE energy')
    ax1.grid()
    ax1.legend()
    plt.legend()
    plt.savefig('results/optimum/tradeoff.png')
    plt.savefig('results/optimum/tradeoff.svg')
    #plt.show()

def plot_kpis_all_episodes(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list,
                           df_energy_credit_list, df_top1_list, df_y_net_list, df_flops_off_list, algorithms,
                           n_episodes_to_train, total_episodes_train, barplot):
    fig, ax = plt.subplots(layout='constrained')
    r = np.arange(len(algorithms))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0
    bar_colors = ['#072140', '#114584', '#9ABCE4']
    # save episode means in a separate df
    # first the optimum
    alg_idx = 0
    df_optimum_inference_time = df_inference_time_list[alg_idx]
    df_optimum_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_optimum_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_optimum_energy_credit = df_energy_credit_list[alg_idx]
    df_optimum_top1 = df_top1_list[alg_idx]
    # then calculate the means
    df_optimum_inference_time['mean'] = df_optimum_inference_time.mean(axis=1)  # row-wise mean
    mean_inference_time_optimum = df_optimum_inference_time['mean'].mean()
    df_optimum_ue_energy_comp['mean'] = df_optimum_ue_energy_comp.mean(axis=1)
    mean_ue_energy_comp_optimum = df_optimum_ue_energy_comp['mean'].mean()
    df_optimum_ue_energy_comm['mean'] = df_optimum_ue_energy_comm.mean(axis=1)
    mean_ue_energy_comm_optimum = df_optimum_ue_energy_comm['mean'].mean()
    mean_ue_energy_optimum = mean_ue_energy_comp_optimum + mean_ue_energy_comm_optimum
    df_optimum_energy_credit['mean'] = df_optimum_energy_credit.mean(axis=1)
    mean_energy_credit_optimum = df_optimum_energy_credit['mean'].mean()
    df_optimum_top1['mean'] = df_optimum_top1.mean(axis=1)
    mean_top1_optimum = df_optimum_top1['mean'].mean()
    # then rl
    alg_idx = 1
    n_episodes_bef_train = n_episodes_to_train
    n_episodes_aft_train = total_episodes_train - n_episodes_bef_train
    df_ddqn_inference_time = df_inference_time_list[alg_idx]
    df_ddqn_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_ddqn_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_ddqn_energy_credit = df_energy_credit_list[alg_idx]
    df_ddqn_top1 = df_top1_list[alg_idx]
    df_ddqn_y_net = df_y_net_list[alg_idx]
    #df_ddqn_flops_off = df_flops_off_list[alg_idx]
    # then calculate the means
    df_ddqn_inference_time['mean'] = df_ddqn_inference_time.mean(axis=1)
    mean_inference_time_ddqn = df_ddqn_inference_time['mean'][n_episodes_aft_train:total_episodes_train].mean()
    df_ddqn_ue_energy_comp['mean'] = df_ddqn_ue_energy_comp.mean(axis=1)
    mean_ue_energy_comp_ddqn = df_ddqn_ue_energy_comp['mean'][n_episodes_aft_train:total_episodes_train].mean()
    df_ddqn_ue_energy_comm['mean'] = df_ddqn_ue_energy_comm.mean(axis=1)
    mean_ue_energy_comm_ddqn = df_ddqn_ue_energy_comm['mean'][n_episodes_aft_train:total_episodes_train].mean()
    mean_ue_energy_ddqn = mean_ue_energy_comp_ddqn + mean_ue_energy_comm_ddqn
    df_ddqn_energy_credit['mean'] = df_ddqn_energy_credit.mean(axis=1)
    mean_energy_credit_ddqn = df_ddqn_energy_credit['mean'][n_episodes_aft_train:total_episodes_train].mean()
    df_ddqn_top1['mean'] = df_ddqn_top1.mean(axis=1)
    mean_top1_ddqn = df_ddqn_top1['mean'][n_episodes_aft_train:total_episodes_train].mean()
    df_ddqn_y_net['mean'] = df_ddqn_y_net.mean(axis=1)
    #df_ddqn_flops_off['mean'] = df_ddqn_flops_off.mean(axis=1)

    # then random
    alg_idx = 2
    df_random_inference_time = df_inference_time_list[alg_idx]
    df_random_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_random_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_random_energy_credit = df_energy_credit_list[alg_idx]
    df_random_top1 = df_top1_list[alg_idx]
    # then calculate the means
    df_random_inference_time['mean'] = df_random_inference_time.mean(axis=1)
    mean_inference_time_random = df_random_inference_time['mean'].mean()
    df_random_ue_energy_comp['mean'] = df_random_ue_energy_comp.mean(axis=1)
    mean_ue_energy_comp_random = df_random_ue_energy_comp['mean'].mean()
    df_random_ue_energy_comm['mean'] = df_random_ue_energy_comm.mean(axis=1)
    mean_ue_energy_comm_random = df_random_ue_energy_comm['mean'].mean()
    mean_ue_energy_random = mean_ue_energy_comp_random + mean_ue_energy_comm_random
    df_random_energy_credit['mean'] = df_random_energy_credit.mean(axis=1)
    mean_energy_credit_random = df_random_energy_credit['mean'].mean()
    df_random_top1['mean'] = df_random_top1.mean(axis=1)
    mean_top1_random = df_random_top1['mean'].mean()


    # then fixed split
    alg_idx = 3
    df_fixed_inference_time = df_inference_time_list[alg_idx]
    df_fixed_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_fixed_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_fixed_energy_credit = df_energy_credit_list[alg_idx]
    df_fixed_top1 = df_top1_list[alg_idx]
    # then calculate the means
    df_fixed_inference_time['mean'] = df_fixed_inference_time.mean(axis=1)
    mean_inference_time_fixed = df_fixed_inference_time['mean'].mean()
    df_fixed_ue_energy_comp['mean'] = df_fixed_ue_energy_comp.mean(axis=1)
    mean_ue_energy_comp_fixed = df_fixed_ue_energy_comp['mean'].mean()
    df_fixed_ue_energy_comm['mean'] = df_fixed_ue_energy_comm.mean(axis=1)
    mean_ue_energy_comm_fixed = df_fixed_ue_energy_comm['mean'].mean()
    mean_ue_energy_fixed = mean_ue_energy_comp_fixed + mean_ue_energy_comm_fixed
    df_fixed_energy_credit['mean'] = df_fixed_energy_credit.mean(axis=1)
    mean_energy_credit_fixed = df_fixed_energy_credit['mean'].mean()
    df_fixed_top1['mean'] = df_fixed_top1.mean(axis=1)
    mean_top1_fixed = df_fixed_top1['mean'].mean()
    #print(df_fixed_inference_time['mean'])
    # then ue only i.e. local computation
    alg_idx = 4
    df_local_inference_time = df_inference_time_list[alg_idx]
    df_local_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_local_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_local_energy_credit = df_energy_credit_list[alg_idx]
    df_local_top1 = df_top1_list[alg_idx]
    # then calculate the means
    df_local_inference_time['mean'] = df_local_inference_time.mean(axis=1)
    mean_inference_time_local = df_local_inference_time['mean'].mean()
    df_local_ue_energy_comp['mean'] = df_local_ue_energy_comp.mean(axis=1)
    mean_ue_energy_comp_local = df_local_ue_energy_comp['mean'].mean()
    df_local_ue_energy_comm['mean'] = df_local_ue_energy_comm.mean(axis=1)
    mean_ue_energy_comm_local = df_local_ue_energy_comm['mean'].mean()
    mean_ue_energy_local = mean_ue_energy_comp_local + mean_ue_energy_comm_local
    df_local_energy_credit['mean'] = df_local_energy_credit.mean(axis=1)
    mean_energy_credit_local = df_local_energy_credit['mean'].mean()
    df_local_top1['mean'] = df_local_top1.mean(axis=1)
    mean_top1_local = df_local_top1['mean'].mean()
    data = {}
    # gather in a dict
    data = {'inference_latency': (mean_inference_time_optimum, mean_inference_time_ddqn, mean_inference_time_random,
                                  mean_inference_time_fixed, mean_inference_time_local),
            'ue_energy_comp': (mean_ue_energy_comp_optimum, mean_ue_energy_comp_ddqn, mean_ue_energy_comp_random,
                               mean_ue_energy_comp_fixed, mean_ue_energy_comp_local),
            'ue_energy_comm': (mean_ue_energy_comm_optimum, mean_ue_energy_comm_ddqn, mean_ue_energy_comm_random,
                               mean_ue_energy_comm_fixed, mean_ue_energy_comm_local),
            'mean_ue_energy': (mean_ue_energy_optimum, mean_ue_energy_ddqn, mean_ue_energy_random,
                               mean_ue_energy_fixed, mean_ue_energy_local),
            'top1': (mean_top1_optimum, mean_top1_ddqn, mean_top1_random, mean_top1_fixed, mean_top1_local)}

    # check if the plot to generate is a barplot, otherwise generate a usual line/scatter plot
    if barplot:
        offset = width * multiplier
        rects = ax.bar(r + offset, data['inference_latency'], width=width, color='black', label='inference_latency')
        ax.bar_label(rects, padding=3, fontsize=8)
        #print('Mean Inference latency {}'.format(rects))
        multiplier += 1
        offset = width * multiplier
        rects = ax.bar(r + offset, data['ue_energy_comp'], width=width, color='#072140', label='ue_energy_comp')
        ax.bar_label(rects, padding=3, fontsize=8)
        #print('Mean UE energy comp {}'.format(rects))
        multiplier += 1
        offset = width * multiplier
        rects = ax.bar(r + offset, data['ue_energy_comm'], width=width, color='#165DB1', label='ue_energy_comm')
        ax.bar_label(rects, padding=3, fontsize=8)
        #print('Mean UE energy comm {}'.format(rects))
        multiplier += 1
        offset = width * multiplier
        rects = ax.bar(r + offset, data['mean_ue_energy'], width=width, color='#9ABCE4', label='sum_ue_energy')
        ax.bar_label(rects, padding=3, fontsize=8)

        multiplier += 1
        offset = width * multiplier
        rects = ax.bar(r + offset, data['top1'], width=width, color='blue', label='top1')
        ax.bar_label(rects, padding=3, fontsize=8)

        ax.set_ylabel('Log value')
        ax.set_xticks(r + width, algorithms)
        ax.legend(loc='upper left', ncols=1)
        ax.grid()
        plt.yscale('log')
        plt.savefig('results/inference_energy_comparison_mobility.png')
        plt.savefig('results/inference_energy_comparison_mobility.svg')
        plt.show()
    else:
        window = 1
        # ax.scatter(df_ddqn_ue_energy_comm['mean'][n_episodes_aft_train:].rolling(window=window).mean(),
        #            df_ddqn_energy_credit['mean'][n_episodes_aft_train:].rolling(window=window).mean(),
        #            marker='^', color='#072140', label='ddqn')
        x = df_ddqn_inference_time['mean'].iloc[:n_episodes_aft_train]
        #x1 = df_ddqn_ue_energy_comm['mean'].iloc[:n_episodes_aft_train]
        #x2 = df_ddqn_ue_energy_comp['mean'].iloc[:n_episodes_aft_train]
        #x1 = df_ddqn_flops_off['mean'].iloc[:n_episodes_aft_train]
        x2 = df_ddqn_y_net['mean'].iloc[:n_episodes_aft_train]
        #x = x1
        #x = x1 + x2
        #y = df_ddqn_ue_energy_comm['mean'].iloc[:n_episodes_aft_train]
        y = df_ddqn_ue_energy_comp['mean'].iloc[:n_episodes_aft_train]
        ax.scatter(x, y, marker='^', color='#072140', label='ddqn')
        x = mean_inference_time_optimum
        y = mean_ue_energy_comp_optimum
        plt.axhline(y=y)
        plt.axvline(x=x)
        #print(df_optimum_inference_time)
        #ax.scatter(x, y, marker='^', color='#9ABCE4', label='optimum')
        x = df_random_inference_time['mean']
        y = df_random_ue_energy_comp['mean']
        #ax.scatter(x, y, marker='o', color='#165DB1', label='random')
        x = df_fixed_inference_time
        y = df_fixed_ue_energy_comm
        #ax.scatter(x, y, marker='^', label='fixed')
        #print(df_ddqn_ue_energy_comm['mean'].iloc[:n_episodes_aft_train])
        #print(df_ddqn_energy_credit['mean'][:n_episodes_aft_train])
        #ax2 = ax.twinx()

        #b, a = np.polyfit(x, y, deg=1)
        #xseq = np.linspace(0.8, 0.875, num=10)
        #ax.plot(xseq, a + b * xseq, color="k", lw=2.5)
        #ax.scatter(df_random_ue_energy_comm['mean'], df_random_energy_credit['mean'], color='#165DB1', marker='o', label='random')
        #print(df_ddqn_inference_time['mean'][:n_episodes_aft_train].rolling(window=window).mean())
        #print(df_ddqn_energy_credit['mean'][:n_episodes_aft_train].rolling(window=window).mean())
        ax.grid()
        #plt.axhline(y=mean_energy_credit_optimum, color='black', linestyle='-', linewidth='2')
        #plt.axvline(x=mean_ue_energy_comm_optimum, color='blue', linestyle='--', linewidth='2')
        ax.set_xlabel('inference latency')
        ax.set_ylabel('ue energy comp')
        #ax.legend()
        plt.legend()
        plt.savefig('results/rl/ddqn/latency_energy_comp.png')
        plt.savefig('results/rl/ddqn/latency_energy_comp.svg')
        plt.show()


def plot_kpis(df_list, n_episodes, n_episodes_to_plot, algorithms, kpi_type):
    """
    Generates an error plot of a given system kpi over episodes.

    Args:
        df_list (list of pandas DataFrame): list of 2D dataframes containing of kpi logs across and within episodes
        for algorithms
        n_episodes (dict): number of episodes
        n_episodes_to_plot (int): the actual number of episodes to plot
        algorithms (list): Indicates the algorithms to plot e.g. random/rl/optimum.
        kpi_type (str): the kpi to plot e.g. inference_time

    Returns:

    """
    fig, ax = plt.subplots()
    plt.axhline(y=0.9, color='red', linestyle='--', linewidth='2', label='energy credit limit')
    #plt.axvline(x=0.9, color='red', linestyle='--', linewidth='2', label='energy credit limit')
    colors = {'random': 'D', 'rl/ddqn': 'o'}
    episodes = [ep for ep in range(1, n_episodes_to_plot + 1)]
    # start extracting data and plotting for each algorithm
    for i, alg in enumerate(algorithms):
        df = df_list[i]
        print(df)
        mean_per_episode = []
        sd_per_episode = []
        yerr_per_episode = []

        for ep in range(1, n_episodes[alg] + 1):
            mean_per_episode.append(df.loc[ep].mean())
            sd = df.loc[ep].std()
            sd_per_episode.append(sd)
            yerr_per_episode.append(get_confidence_interval(sd, n_episodes[alg]))

        if alg == 'optimum':
            plt.axhline(y=mean_per_episode[0], color='black', linestyle='-', linewidth='5', label='optimum')
            #plt.axvline(x=mean_per_episode[0], color='black', linestyle='-', linewidth='5', label='optimum')
        else:
            ax.errorbar(episodes, mean_per_episode, yerr=yerr_per_episode, fmt=colors[alg], ecolor='black',
                     label='{}'.format(alg))
            #plt.hist(mean_per_episode, alpha=0.5, label='{}'.format(alg))
    #ax.set_xlabel('Energy credit usage')
    ax.set_xlabel('Episode')
    #ax.set_ylabel('Number of episodes')
    ax.set_ylabel('Energy credit usage')
    ax.set_ylim(top=0.95)

    plt.grid()
    plt.legend()
    #plt.savefig('results/{}.png'.format(kpi_type))
    #plt.savefig('results/{}.svg'.format(kpi_type))
    plt.show()

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
    if folder != 'rl/ddqn':
        order = return_order(n_episodes)
    else:
        order = return_order(n_episodes=1000)
    inference_times_all_episodes = []
    ue_energy_comp_all_episodes = []
    ue_energy_comm_all_episodes = []
    energy_credit_all_episodes = []
    top1_all_episodes = []
    y_net_all_episodes = []
    flops_off_all_episodes = []
    time_steps = []

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
        #time_steps, energy_credit_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
        #energy_credit_all_episodes.append(energy_credit_per_episode)
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
    df_top1 = pd.DataFrame(top1_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_y_net = pd.DataFrame(y_net_all_episodes, columns=time_steps,
                                    index=[ep for ep in range(1, n_episodes + 1)])
    df_flops_off = pd.DataFrame(flops_off_all_episodes, columns=time_steps,
                                    index=[ep for ep in range(1, n_episodes + 1)])

    return df_inference_time, df_ue_energy_comp, df_ue_energy_comm, df_energy_credit, df_top1, df_y_net, df_flops_off

def parse_kpis_optimum(n_episodes, omega):
    """
    Function to read and parse kpis of the optimal solution for a specified value of omega into a 2D pandas DataFrame.
    Args:
        n_episodes (int): number of episodes to parse (determines the order)
        omega (float): the specified value of omega.

    Returns:
        Tuple: (dataframes containing inference time, ue computation and communication energy)
    """
    order = return_order(n_episodes)
    inference_times_all_episodes = []
    ue_energy_comp_all_episodes = []
    ue_energy_comm_all_episodes = []
    time_steps = []
    folder = 'optimum/omega_{}'.format(omega)
    for episode in range(1, n_episodes + 1):
        episode_count = parse_episode_number(order, episode)
        kpi_type = 'inference_time'
        time_steps, inference_times_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, None)
        inference_times_all_episodes.append(inference_times_per_episode)
        kpi_type = 'ue_energy_comp'
        time_steps, ue_energy_comp_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, None)
        ue_energy_comp_all_episodes.append(ue_energy_comp_per_episode)
        kpi_type = 'ue_energy_comm'
        time_steps, ue_energy_comm_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, None)
        ue_energy_comm_all_episodes.append(ue_energy_comm_per_episode)

    # concatenate data of all episodes into single data structure
    df_inference_time = pd.DataFrame(inference_times_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_ue_energy_comp = pd.DataFrame(ue_energy_comp_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_ue_energy_comm = pd.DataFrame(ue_energy_comm_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])

    return df_inference_time, df_ue_energy_comp, df_ue_energy_comm

def main():
    """
    Main function that invokes other functions to read and parse kpis and plot them.

    Returns:

    """
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    n_episodes_to_plot = 1500
    #n_episodes = {'optimum': 9, 'random': 200, 'rl/ddqn': 5000}
    n_episodes = {'optimum': 9, 'rl/ddqn': 5000, 'random': 200, 'fixed': 9, 'ue': 1}
    #n_episodes = {'optimum': 1, 'rl/ddqn': 1500, 'random': 15, 'fixed': 1}
    algorithms = ['optimum', 'rl/ddqn', 'random', 'fixed', 'ue']
    #algorithms = ['optimum', 'rl/ddqn', 'random', 'fixed']
    #algorithms = ['optimum', 'rl/ddqn', 'random']
    #algorithms = ['rl/ddqn']
    df_inference_time_list = [] # for each specified algorithm in 'algorithms'
    df_ue_energy_comp_list = [] # for each specified algorithm in 'algorithms'
    df_ue_energy_comm_list = [] # for each specified algorithm in 'algorithms'
    df_energy_credit_list = []  # for each specified algorithm in 'algorithms'
    df_top1_list = []   # for each specified algorithm in 'algorithms'
    df_y_net_list = []  # for each specified algorithm in 'algorithms'
    df_flops_off_list = []  # for each specified algorithm in 'algorithms'

    # specifies the episodes of convergence of ddqn
    n_episodes_to_train = 900   # mean inference latency & ue energy of different methods
    n_episodes_to_train = 2000  # for inference latency vs energy plots
    total_episodes_train = 5000 # mean inference latency & ue energy of different methods
    for alg in algorithms:
        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, df_energy_credit, df_top1, df_y_net, df_flops_off = (
            parse_kpis(alg, n_episodes[alg], None))
        df_inference_time_list.append(df_inference_time)
        df_ue_energy_comp_list.append(df_ue_energy_comp)
        df_ue_energy_comm_list.append(df_ue_energy_comm)
        df_energy_credit_list.append(df_energy_credit)
        df_top1_list.append(df_top1)
        df_y_net_list.append(df_y_net)
        df_flops_off_list.append(df_flops_off)
    # plot the required kpis across episodes
    #plot_kpis(df_inference_time_list, n_episodes, n_episodes_to_plot, algorithms, 'inference_time')
    #plot_kpis(df_ue_energy_comp_list, n_episodes, n_episodes_to_plot, algorithms, 'ue_energy_comp')
    #plot_kpis(df_ue_energy_comm_list, n_episodes, n_episodes_to_plot, algorithms, 'ue_energy_comm')
    #plot_kpis(df_energy_credit_list, n_episodes, n_episodes_to_plot, algorithms, 'energy_credit')

    # debug

    # barplot=False implies plotting one kpi vs another e.g. energy credit vs ue energy
    plot_kpis_all_episodes(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list,
    df_energy_credit_list, df_top1_list, df_y_net_list, df_flops_off_list, algorithms, n_episodes_to_train, total_episodes_train,
                           barplot=True)

    # -------------------------- Only for optimum ---------------------------
    omega_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_episodes_opt = 1
    df_inference_time_list = []  # for each specified omega in 'omega_list'
    df_ue_energy_comp_list = []  # for each specified omega in 'omega_list'
    df_ue_energy_comm_list = []  # for each specified omega in 'omega_list'

    # for omega in omega_list:
    #     df_inference_time, df_ue_energy_comp, df_ue_energy_comm = parse_kpis_optimum(n_episodes_opt, omega)
    #     df_inference_time_list.append(df_inference_time)
    #     df_ue_energy_comp_list.append(df_ue_energy_comp)
    #     df_ue_energy_comm_list.append(df_ue_energy_comm)

    # only one episode is sufficient
    #plot_kpis_tradeoff_optimum(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list, omega_list)
    # -------------------------- Only for optimum ---------------------------


if __name__ == '__main__':
    main()