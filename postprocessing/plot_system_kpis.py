import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.logging_utils import return_order, parse_episode_number


def read_kpis_from_files(folder, kpi_type, episode_count):
    """
    Function to read system KPIs (inference time, ue computation and communication energy) for each episode.

    Args:
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        kpi_type (str): The kpi to read e.g. inference_time
        episode_count (int): The episode number

    Returns:
        Tuple: (time step when kpi was recorded as list, kpi as list )
    """
    file = 'logs/{}/system/{}_{}.csv'.format(folder, kpi_type, episode_count)
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

def plot_kpis_tradeoff_algorithms(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list, n_episodes,
                       n_episodes_to_plot, omega_list, algorithms):
    pass

def plot_kpis_tradeoff_optimum(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list, n_episodes,
                       n_episodes_to_plot, omega_list):
    """
    Script to plot the tradeoff curve between inference time and ue energy for the optimum case for different values of
    the weight, omega. This tradeoff should be analysed using static values of the system parameters.
    Args:
        df_inference_time_list (list of pandas DataFrame): list of 2D dataframes containing of inference time logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_ue_energy_comp_list (list of pandas DataFrame): list of 2D dataframes containing of ue_energy_comp logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        df_ue_energy_comm_list (list of pandas DataFrame): list of 2D dataframes containing of ue_energy_comm logs across and
        within episodes. N_rows = n_episodes while N_columns = n_timesteps.
        n_episodes (dict): number of episodes
        n_episodes_to_plot (int): the actual number of episodes to plot
        omega_list (list): list of values of omega to plot.

    Returns:

    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 0.7, n_episodes_to_plot))
    for i, omega in enumerate(omega_list):
        df_inference_time = df_inference_time_list[i]
        df_ue_energy_comp = df_ue_energy_comp_list[i]
        df_ue_energy_comm = df_ue_energy_comm_list[i]
        print(df_inference_time)
        print(df_ue_energy_comp)
        print(df_ue_energy_comm)
        for ep in range(n_episodes_to_plot):
            sum_ue_energies = df_ue_energy_comp.iloc[ep] + df_ue_energy_comm.iloc[ep]
            plt.bar(omega * df_inference_time.iloc[ep], (1 - omega) * sum_ue_energies, width=0.001, color=colors)
    ax.grid()
    ax.set_xlabel('weighted inference time in s')
    ax.set_ylabel('weighted sum of ue energy in J')
    plt.show()

def plot_kpis_all_episodes(df_list, n_episodes, n_episodes_to_plot, algorithms, kpi_type, omega_list, n_episodes_to_train,
                           total_episodes_train):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 0.7, len(algorithms)))
    position = 0
    # save episode means in a separate df
    for i, omega in enumerate(omega_list):
        position = position + 5
        # first the optimum
        alg_idx = 0
        algorithm = 'optimum'
        df_optimum = df_list[alg_idx]
        mean_per_episode = []
        for ep in range(1, n_episodes[algorithm] + 1):
            mean_per_episode.append(df_optimum.loc[ep].mean())
        df_optimum_means = pd.DataFrame(mean_per_episode)
        mean = df_optimum_means.mean()
        sd = df_optimum_means.std()
        h = get_confidence_interval(sd, n_episodes[algorithm])
        ax.errorbar(position, mean, yerr=h, label='optimum')
        # then rl
        position = position + 5
        alg_idx = 1
        df_ddqn = df_list[alg_idx]
        #print(df_ddqn)
        mean_per_episode = []
        n_episodes_bef_train = n_episodes_to_train[omega]
        n_episodes_aft_train = total_episodes_train[omega] - n_episodes_bef_train
        #print(n_episodes_aft_train)
        for ep in range(n_episodes_bef_train, total_episodes_train[omega] + 1):
            mean_per_episode.append(df_ddqn.loc[ep].mean())
        df_ddqn_means = pd.DataFrame(mean_per_episode)
        #print(df_ddqn_means)
        mean = df_ddqn_means.mean()
        #print(mean)
        sd = df_ddqn_means.std()
        #print(sd)
        h = get_confidence_interval(sd, n_episodes_aft_train)
        ax.errorbar(position, mean, yerr=h, label='ddqn')

    ax.grid()
    plt.legend()
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
    colors = {'random': 'r--o', 'rl/ddqn': 'b--o'}
    episodes = [ep for ep in range(1, n_episodes_to_plot + 1)]
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
            plt.axhline(y=mean_per_episode[0], color='g', linestyle='-')
        else:
            ax.errorbar(episodes, mean_per_episode, yerr=yerr_per_episode, fmt=colors[alg], ecolor='black',
                     label='{}'.format(alg))
    ax.set_xlabel('episodes')
    ax.set_ylabel('{}'.format(kpi_type))
    plt.grid()
    plt.legend()
    plt.savefig('results/{}.png'.format(kpi_type))
    plt.savefig('results/{}.svg'.format(kpi_type))
    plt.show()

def parse_kpis(folder, n_episodes):
    """
    Function to read and parse kpis into a 2D pandas DataFrame.
    Args:
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        n_episodes (int): number of episodes per algorithm to print

    Returns:
        Tuple: (dataframes containing inference time, ue computation and communication energy)
    """
    order = return_order(n_episodes)
    inference_times_all_episodes = []
    ue_energy_comp_all_episodes = []
    ue_energy_comm_all_episodes = []
    time_steps = []

    for episode in range(1, n_episodes + 1):
        episode_count = parse_episode_number(order, episode)
        kpi_type = 'inference_time'
        time_steps, inference_times_per_episode = read_kpis_from_files(folder, kpi_type, episode_count)
        inference_times_all_episodes.append(inference_times_per_episode)
        kpi_type = 'ue_energy_comp'
        time_steps, ue_energy_comp_per_episode = read_kpis_from_files(folder, kpi_type, episode_count)
        ue_energy_comp_all_episodes.append(ue_energy_comp_per_episode)
        kpi_type = 'ue_energy_comm'
        time_steps, ue_energy_comm_per_episode = read_kpis_from_files(folder, kpi_type, episode_count)
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

    n_episodes_to_plot = 2
    n_episodes = {'optimum': 1, 'random': 999, 'rl/ddqn': 999}
    n_episodes = {'optimum': 2, 'rl/ddqn': 2380}
    algorithms = ['optimum', 'random', 'rl/ddqn']
    algorithms = ['optimum', 'rl/ddqn']
    #algorithms = ['optimum']
    df_inference_time_list = [] # for each specified algorithm in 'algorithms'
    df_ue_energy_comp_list = [] # for each specified algorithm in 'algorithms'
    df_ue_energy_comm_list = [] # for each specified algorithm in 'algorithms'

    omega_list = [0.1]
    # specifies the episodes of convergence of ddqn for the corresponding value of omega
    n_episodes_to_train = {0.1: 2000, 0.3: 1000, 0.5: 1000, 0.7: 1000, 0.9: 1000}
    total_episodes_train = {0.1: 2380, 0.3: 1000, 0.5: 1000, 0.7: 1000, 0.9: 1000}
    for alg in algorithms:
        df_inference_time, df_ue_energy_comp, df_ue_energy_comm = parse_kpis(alg, n_episodes[alg])
        df_inference_time_list.append(df_inference_time)
        df_ue_energy_comp_list.append(df_ue_energy_comp)
        df_ue_energy_comm_list.append(df_ue_energy_comm)
    # plot the required kpis
    #plot_kpis(df_inference_time_list, n_episodes, n_episodes_to_plot, algorithms, 'inference_time')
    #plot_kpis(df_ue_energy_comp_list, n_episodes, n_episodes_to_plot, algorithms, 'ue_energy_comp')
    #plot_kpis(df_ue_energy_comm_list, n_episodes, n_episodes_to_plot, algorithms, 'ue_energy_comm')

    #plot_kpis_tradeoff_optimum(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list, n_episodes,
    #                           n_episodes_to_plot, omega_list)

    plot_kpis_all_episodes(df_inference_time_list, n_episodes, n_episodes_to_plot, algorithms, 'inference_time',
                           omega_list, n_episodes_to_train, total_episodes_train)

if __name__ == '__main__':
    main()