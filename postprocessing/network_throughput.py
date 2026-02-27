import os, csv
import pandas as pd
import numpy as np
from postprocessing.plot_system_kpis import return_order, parse_episode_number
import matplotlib.pyplot as plt


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

    return df_inference_time, df_ue_energy_comp, df_ue_energy_comm, df_energy_credit, df_y_net


def plot_kpis_vs_max_throughput(df_all_inference_time, df_all_ue_energy_comp, df_all_ue_energy_comm,
                                    max_throughput_list, n_episodes_to_train, total_episodes_train):
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
    # df_inference_time_ddqn_list = df_all_inference_time[0]   # ddqn
    # df_ue_energy_comp_ddqn_list = df_all_ue_energy_comp[0]   # ddqn
    # df_ue_energy_comm_ddqn_list = df_all_ue_energy_comm[0]   # ddqn
    df_inference_time_opt_list = df_all_inference_time[1]   # opt
    df_ue_energy_comp_opt_list = df_all_ue_energy_comp[1]   # opt
    df_ue_energy_comm_opt_list = df_all_ue_energy_comm[1]   # opt
    df_inference_time_random_list = df_all_inference_time[2]    # random
    df_ue_energy_comp_random_list = df_all_ue_energy_comp[2]    # random
    df_ue_energy_comm_random_list = df_all_ue_energy_comm[2]    # random
    # df_inference_time_fixed_list = df_all_inference_time[3]     # fixed
    # df_ue_energy_comp_fixed_list = df_all_ue_energy_comp[3]     # fixed
    # df_ue_energy_comm_fixed_list = df_all_ue_energy_comm[3]     # fixed

    n_episodes_bef_train = n_episodes_to_train
    n_episodes_aft_train = total_episodes_train - n_episodes_bef_train
    fig, ax = plt.subplots()
    # inference_time_mean_per_deadline_ddqn = []
    # ue_energy_comp_mean_per_deadline_ddqn = []
    # ue_energy_comm_mean_per_deadline_ddqn = []
    inference_time_mean_per_deadline_opt = []
    ue_energy_comp_mean_per_deadline_opt = []
    ue_energy_comm_mean_per_deadline_opt = []
    inference_time_mean_per_deadline_random = []
    ue_energy_comp_mean_per_deadline_random = []
    ue_energy_comm_mean_per_deadline_random = []
    # inference_time_mean_per_deadline_fixed = []
    # ue_energy_comp_mean_per_deadline_fixed = []
    # ue_energy_comm_mean_per_deadline_fixed = []
    # extract data and store means
    for i, max_throughput in enumerate(max_throughput_list):
        # df_inference_time_ddqn = df_inference_time_ddqn_list[i]
        # df_ue_energy_comp_ddqn = df_ue_energy_comp_ddqn_list[i]
        # df_ue_energy_comm_ddqn = df_ue_energy_comm_ddqn_list[i]
        # # then calculate the means
        # df_inference_time_ddqn['mean'] = df_inference_time_ddqn.mean(axis=1)
        # inference_time_mean_per_deadline_ddqn.append(df_inference_time_ddqn['mean'].iloc[n_episodes_bef_train:total_episodes_train].mean())
        # df_ue_energy_comp_ddqn['mean'] = df_ue_energy_comp_ddqn.mean(axis=1)
        # ue_energy_comp_mean_per_deadline_ddqn.append(df_ue_energy_comp_ddqn['mean'].iloc[n_episodes_bef_train:total_episodes_train].mean())
        # df_ue_energy_comm_ddqn['mean'] = df_ue_energy_comm_ddqn.mean(axis=1)
        # ue_energy_comm_mean_per_deadline_ddqn.append(df_ue_energy_comm_ddqn['mean'].iloc[n_episodes_bef_train:total_episodes_train].mean())
        # sum_ue_energy_per_deadline.append(df_ue_energy_comp_ddqn['mean'][n_episodes_bef_train:].mean() +
        #                                   df_ue_energy_comm_ddqn['mean'][n_episodes_bef_train:])
        # then optimum
        df_inference_time_opt = df_inference_time_opt_list[i]
        df_ue_energy_comp_opt = df_ue_energy_comp_opt_list[i]
        df_ue_energy_comm_opt = df_ue_energy_comm_opt_list[i]
        df_inference_time_opt['mean'] = df_inference_time_opt.mean(axis=1)
        inference_time_mean_per_deadline_opt.append(df_inference_time_opt['mean'].mean())
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
        df_ue_energy_comp_random['mean'] = df_ue_energy_comp_random.mean(axis=1)
        ue_energy_comp_mean_per_deadline_random.append(df_ue_energy_comp_random['mean'].mean())
        df_ue_energy_comm_random['mean'] = df_ue_energy_comm_random.mean(axis=1)
        ue_energy_comm_mean_per_deadline_random.append(df_ue_energy_comm_random['mean'].mean())

        # then fixed
        # df_inference_time_fixed = df_inference_time_fixed_list[i]
        # df_ue_energy_comp_fixed = df_ue_energy_comp_fixed_list[i]
        # df_ue_energy_comm_fixed = df_ue_energy_comm_fixed_list[i]
        # df_inference_time_fixed['mean'] = df_inference_time_fixed.mean(axis=1)
        # inference_time_mean_per_deadline_fixed.append(df_inference_time_fixed['mean'].mean())
        # df_ue_energy_comp_fixed['mean'] = df_ue_energy_comp_fixed.mean(axis=1)
        # ue_energy_comp_mean_per_deadline_fixed.append(df_ue_energy_comp_fixed['mean'].mean())
        # df_ue_energy_comm_fixed['mean'] = df_ue_energy_comm_fixed.mean(axis=1)
        # ue_energy_comm_mean_per_deadline_fixed.append(df_ue_energy_comm_fixed['mean'].mean())

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
    ax.plot(max_throughput_list, inference_time_mean_per_deadline_opt, color='#072140', marker='o', label='opt')
    #ax.plot(max_throughput_list, inference_time_mean_per_deadline_ddqn, color='#165DB1', marker='o', label='ddqn')
    ax.plot(max_throughput_list, inference_time_mean_per_deadline_random, color='#9ABCE4', marker='o', label='random')
    #ax.plot(max_throughput_list, inference_time_mean_per_deadline_fixed, color='#8F81EA', marker='o', label='fixed')
    ax.set_xlabel('Network throughput (MB/s)')
    ax.set_ylabel('Inference time (s)')
    #ax.set_ylabel('UE energy comp (J)')
    #ax.set_ylabel('UE energy comm (J)')
    #ax.set_ylabel('UE energy sum (J)')
    #plt.yscale('log')
    plt.grid()
    plt.legend()
    #plt.savefig('results/inference_time_vs_max_throughput.png')
    #plt.savefig('results/inference_time_vs_max_throughput.svg')
    plt.show()


def main():
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    max_network_throughput_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]   # in MB/s
    # specifies the episodes of convergence of ddqn
    n_episodes_to_train = 950
    total_episodes_train = 1000
    df_inference_time_list_ddqn = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_ddqn = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_ddqn = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_opt = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_opt = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_opt = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_random = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_random = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_random = []  # for each specified max throughput in 'max_network_throughput_list'
    df_inference_time_list_fixed = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comp_list_fixed = []  # for each specified max throughput in 'max_network_throughput_list'
    df_ue_energy_comm_list_fixed = []  # for each specified max throughput in 'max_network_throughput_list'

    for max_throughput in max_network_throughput_list:
        # df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _ = parse_kpis('rl/ddqn', total_episodes_train,
        #                                                                      max_throughput)
        # df_inference_time_list_ddqn.append(df_inference_time)
        # df_ue_energy_comp_list_ddqn.append(df_ue_energy_comp)
        # df_ue_energy_comm_list_ddqn.append(df_ue_energy_comm)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _ = parse_kpis('optimum', 9, max_throughput)
        df_inference_time_list_opt.append(df_inference_time)
        df_ue_energy_comp_list_opt.append(df_ue_energy_comp)
        df_ue_energy_comm_list_opt.append(df_ue_energy_comm)

        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _ = parse_kpis('random', 300, max_throughput)
        df_inference_time_list_random.append(df_inference_time)
        df_ue_energy_comp_list_random.append(df_ue_energy_comp)
        df_ue_energy_comm_list_random.append(df_ue_energy_comm)
        #
        # df_inference_time, df_ue_energy_comp, df_ue_energy_comm, _, _ = parse_kpis('fixed', 9, max_throughput)
        # df_inference_time_list_fixed.append(df_inference_time)
        # df_ue_energy_comp_list_fixed.append(df_ue_energy_comp)
        # df_ue_energy_comm_list_fixed.append(df_ue_energy_comm)

    df_all_inference_time = [df_inference_time_list_ddqn, df_inference_time_list_opt, df_inference_time_list_random,
                             df_inference_time_list_fixed]
    df_all_ue_energy_comp = [df_ue_energy_comp_list_ddqn, df_ue_energy_comp_list_opt, df_ue_energy_comp_list_random,
                             df_ue_energy_comp_list_fixed]
    df_all_ue_energy_comm = [df_ue_energy_comm_list_ddqn, df_ue_energy_comm_list_opt, df_ue_energy_comm_list_random,
                             df_ue_energy_comm_list_fixed]
    plot_kpis_vs_max_throughput(df_all_inference_time, df_all_ue_energy_comp, df_all_ue_energy_comm,
                                    max_network_throughput_list, n_episodes_to_train, total_episodes_train)

if __name__ == '__main__':
    main()