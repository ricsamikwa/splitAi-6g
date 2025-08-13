import os
import csv
import pandas as pd
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
    file = 'logs/{}/{}_{}.csv'.format(folder, kpi_type, episode_count)
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

def plot_kpis(df, n_episodes, folder, kpi_type):
    """
    Generates an error plot of a given system kpi over episodes.

    Args:
        df (pandas DataFrame): 2D dataframe containing of kpi logs across and within episodes
        n_episodes (int): number of episodes
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        kpi_type (str): the kpi to plot e.g. inference_time

    Returns:

    """
    fig, ax = plt.subplots()
    mean_per_episode = []
    sd_per_episode = []
    yerr_per_episode = []

    for ep in range(1, n_episodes + 1):
        mean_per_episode.append(df.loc[ep].mean())
        sd = df.loc[ep].std()
        sd_per_episode.append(sd)
        yerr_per_episode.append(get_confidence_interval(sd, n_episodes))

    episodes = [ep for ep in range(1, n_episodes + 1)]
    plt.errorbar(episodes, mean_per_episode, yerr=yerr_per_episode, capsize=3, fmt="r--o", ecolor='black',
                 label='{}'.format(folder))
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
        n_episodes (int): number of episodes

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

    n_episodes = 10
    folder = 'random'
    df_inference_time, df_ue_energy_comp, df_ue_energy_comm = parse_kpis(folder, n_episodes)
    # plot the required kpis
    plot_kpis(df_inference_time, n_episodes, folder, 'inference_time')
    plot_kpis(df_ue_energy_comp, n_episodes, folder, 'ue_energy_comp')
    plot_kpis(df_ue_energy_comm, n_episodes, folder, 'ue_energy_comm')

if __name__ == '__main__':
    main()