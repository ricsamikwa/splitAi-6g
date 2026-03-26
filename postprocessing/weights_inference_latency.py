import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.logging_utils import return_order, parse_episode_number


def pareto_front(latency, energy):
    points = np.array(list(zip(latency, energy)))

    # Sort by latency (ascending)
    points = points[points[:, 0].argsort()]

    pareto = []
    min_energy = float('inf')

    for l, e in points:
        if e < min_energy:
            pareto.append((l, e))
            min_energy = e

    return np.array(pareto)

def read_kpis_from_files(folder, kpi_type, episode_count, weight_inference_latency):
    """
    Function to read system KPIs (inference time, ue computation and communication energy) for each episode.

    Args:
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        kpi_type (str): The kpi to read e.g. inference_time
        episode_count (int): The episode number
        weight_inference_latency (float): the weight of the inference latency in the optimization objective
    Returns:
        Tuple: (time step when kpi was recorded as list, kpi as list )
    """
    file = 'logs/{}/comparison/omega1_{}/system/{}_{}.csv'.format(folder, weight_inference_latency, kpi_type,
                                                                  episode_count)
    data_timestep = []
    data_kpi = []
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for k, item in enumerate(reader):
            if k != 0:
                data_timestep.append(int(item[0]))
                data_kpi.append(float(item[1]))
    return data_timestep, data_kpi

def parse_kpis(folder, n_episodes, weight_inference_latency):
    """
    Function to read and parse kpis into a 2D pandas DataFrame.
    Args:
        folder (str): Indicates the algorithm e.g. random/rl/optimum.
        n_episodes (int): number of episodes per algorithm to print
        weight_inference_latency (float): the weight of the inference latency in the optimization objective

    Returns:
        Tuple: (dataframes containing inference time, ue computation and communication energy and energy credit)
    """
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
        time_steps, inference_times_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, weight_inference_latency)
        inference_times_all_episodes.append(inference_times_per_episode)
        kpi_type = 'ue_energy_comp'
        time_steps, ue_energy_comp_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, weight_inference_latency)
        ue_energy_comp_all_episodes.append(ue_energy_comp_per_episode)
        kpi_type = 'ue_energy_comm'
        time_steps, ue_energy_comm_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, weight_inference_latency)
        ue_energy_comm_all_episodes.append(ue_energy_comm_per_episode)
        kpi_type = 'energy_credit'
        #time_steps, energy_credit_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, inference_deadline)
        #energy_credit_all_episodes.append(energy_credit_per_episode)
        kpi_type = 'top1'
        time_steps, top1_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, weight_inference_latency)
        top1_all_episodes.append(top1_per_episode)
        if folder == 'rl/ddqn':
            kpi_type = 'y_net'
            time_steps, y_net_per_episode = read_kpis_from_files(folder, kpi_type, episode_count, weight_inference_latency)
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


def plot_kpis(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list,
          df_energy_credit_list, df_top1_list, df_y_net_list, df_flops_off_list, n_episodes_to_train,
          total_episodes_train, inference_latency_weights):
    fig, ax1 = plt.subplots(layout='constrained')
    #fig, axs = plt.subplots(1, len(inference_latency_weights), figsize=(15, 4))
    ax2 = ax1.twinx()
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('weighted inference latency (s)')
    ax2.set_ylabel('weighted ue energy (J)')
    n_episodes_aft_train = total_episodes_train - n_episodes_to_train
    # store data
    inference_time = []
    ue_energy_sum = []
    cmaps = ['Reds', 'Blues', 'Greens']
    for alg_idx in range(3):
        #alg_idx = 0     # for omega1 = 0.1
        weight = inference_latency_weights[alg_idx]
        df_ddqn_inference_time = df_inference_time_list[alg_idx]
        df_ddqn_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
        df_ddqn_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
        df_ddqn_energy_credit = df_energy_credit_list[alg_idx]
        df_ddqn_top1 = df_top1_list[alg_idx]
        df_ddqn_y_net = df_y_net_list[alg_idx]
        # df_ddqn_flops_off = df_flops_off_list[alg_idx]
        # then calculate the means
        df_ddqn_inference_time['mean'] = df_ddqn_inference_time.mean(axis=1)
        #mean_inference_time_ddqn = df_ddqn_inference_time['mean'][n_episodes_aft_train:total_episodes_train].mean()
        df_ddqn_ue_energy_comp['mean'] = df_ddqn_ue_energy_comp.mean(axis=1)
        #mean_ue_energy_comp_ddqn = df_ddqn_ue_energy_comp['mean'][n_episodes_aft_train:total_episodes_train].mean()
        df_ddqn_ue_energy_comm['mean'] = df_ddqn_ue_energy_comm.mean(axis=1)
        #mean_ue_energy_comm_ddqn = df_ddqn_ue_energy_comm['mean'][n_episodes_aft_train:total_episodes_train].mean()
        #mean_ue_energy_ddqn = mean_ue_energy_comp_ddqn + mean_ue_energy_comm_ddqn
        df_ddqn_energy_credit['mean'] = df_ddqn_energy_credit.mean(axis=1)
        #mean_energy_credit_ddqn = df_ddqn_energy_credit['mean'][n_episodes_aft_train:total_episodes_train].mean()
        # df_ddqn_top1['mean'] = df_ddqn_top1.mean(axis=1)
        # mean_top1_ddqn = df_ddqn_top1['mean'][n_episodes_aft_train:total_episodes_train].mean()
        # df_ddqn_y_net['mean'] = df_ddqn_y_net.mean(axis=1)

        # store data here
        inf_time = df_ddqn_inference_time['mean'].iloc[:n_episodes_aft_train].mean()
        comp = df_ddqn_ue_energy_comp['mean'].iloc[:n_episodes_aft_train].mean()
        comm = df_ddqn_ue_energy_comm['mean'].iloc[:n_episodes_aft_train].mean()
        inference_time.append(weight * inf_time)
        ue_energy_overall = comp + comm
        ue_energy_sum.append((1 - weight) * ue_energy_overall)
        objective = weight * np.array(inf_time) + (1 - weight) * np.array(ue_energy_overall)
        #best_idx = np.argmin(objective)
        #inference_time.append(np.array(inf_time)[best_idx])
        #ue_energy_sum.append(np.array(ue_energy_overall)[best_idx])

        #plt.plot(objective, label='w={}'.format(weight))
        #ax1.plot(weight * np.array(inf_time), label='w={}'.format(weight))
        #ax2.plot((1 - weight) * np.array(ue_energy_overall), label='w={}'.format(weight))
        #plt.scatter(inf_time, ue_energy_overall, c=objective, cmap=cmap, label='w={}'.format(weight))
        jitter = 0.01  # adjust based on your scale
        #lat_jitter = np.array(inf_time) + np.random.normal(0, jitter, size=len(inf_time))
        #eng_jitter = np.array(ue_energy_overall) + np.random.normal(0, jitter, size=len(ue_energy_overall))
        #plt.scatter(lat_jitter, eng_jitter, c=objective, cmap='viridis', alpha=0.5)
        #plt.scatter(inf_time, ue_energy_overall, c=objective, cmap='viridis', alpha=0.3)
        #ax.set_title('weight{}'.format(weight))
        #ax.set_xlabel("Latency")
        #ax.set_ylabel("Energy")
    #fig.colorbar(sc, ax=axs, label="Objective")
    ax1.plot(inference_latency_weights, inference_time, color='blue', marker='^', label='inference_latency')
    ax2.plot(inference_latency_weights, ue_energy_sum, color='black', marker='o', label='ue_energy_sum')
    #plt.plot(inference_latency_weights, inference_time, marker='o', label="inference_latency")
    #plt.plot(inference_latency_weights, ue_energy_sum, marker='o', label="ue_energy_sum")
    #ax1.set_xlabel('weight')
    #ax1.set_ylabel('value')
    plt.grid()
    plt.legend()
    plt.show()


def main():
    """
    Main function that invokes other functions to read and parse kpis and plot them.

    Returns:

    """
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    inference_latency_weights = [0.1, 0.5, 0.9]
    df_inference_time_list = [] # for each specified weight in 'inference_latency_weights'
    df_ue_energy_comp_list = [] # for each specified weight in 'inference_latency_weights'
    df_ue_energy_comm_list = [] # for each specified weight in 'inference_latency_weights'
    df_energy_credit_list = []  # for each specified weight in 'inference_latency_weights'
    df_top1_list = []   # for each specified weight in 'inference_latency_weights'
    df_y_net_list = []  # for each specified weight in 'inference_latency_weights'
    df_flops_off_list = []  # for each specified weight in 'inference_latency_weights'

    # specifies the episodes of convergence of ddqn
    n_episodes_to_train = 4800  # for inference latency vs energy plots
    total_episodes_train = 5000 # mean inference latency & ue energy of different methods
    for weight in inference_latency_weights:
        df_inference_time, df_ue_energy_comp, df_ue_energy_comm, df_energy_credit, df_top1, df_y_net, df_flops_off = (
            parse_kpis('rl/ddqn', total_episodes_train, weight))
        df_inference_time_list.append(df_inference_time)
        df_ue_energy_comp_list.append(df_ue_energy_comp)
        df_ue_energy_comm_list.append(df_ue_energy_comm)
        df_energy_credit_list.append(df_energy_credit)
        df_top1_list.append(df_top1)
        df_y_net_list.append(df_y_net)
        df_flops_off_list.append(df_flops_off)

    plot_kpis(df_inference_time_list, df_ue_energy_comp_list, df_ue_energy_comm_list,
    df_energy_credit_list, df_top1_list, df_y_net_list, df_flops_off_list, n_episodes_to_train, total_episodes_train,
                           inference_latency_weights)

if __name__ == '__main__':
    main()