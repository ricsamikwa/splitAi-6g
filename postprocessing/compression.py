import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from utils.logging_utils import return_order, parse_episode_number


def read_kpis_from_files(folder, kpi_type, episode_count):
    if kpi_type == 'top1' or kpi_type == 'ue_energy_comm':
        file = 'logs/{}/system/{}_{}.csv'.format(folder, kpi_type, episode_count)
    else:
        file = 'logs/{}/splits/{}_{}.csv'.format(folder, kpi_type, episode_count)
    data_timestep = []
    data_kpi = []
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for k, item in enumerate(reader):
            if k != 0:
                data_timestep.append(int(item[0]))
                data_kpi.append(float(item[1]))
    return data_timestep, data_kpi

def parse_kpis(alg, n_episodes):
    order = return_order(n_episodes=1000)
    time_steps = []
    compression_all_episodes = []
    top1_accuracy_all_episodes = []
    split_idx_all_episodes = []
    ue_energy_comm_all_episodes = []
    for episode in range(1, n_episodes + 1):
        episode_count = parse_episode_number(order, episode)
        kpi_type = 'compression'
        time_steps, compression_per_episode = read_kpis_from_files(alg, kpi_type, episode_count)
        compression_all_episodes.append(compression_per_episode)
        kpi_type = 'top1'
        time_steps, top1_per_episode = read_kpis_from_files(alg, kpi_type, episode_count)
        top1_accuracy_all_episodes.append(top1_per_episode)
        kpi_type = 'split_idx'
        time_steps, split_idx_per_episode = read_kpis_from_files(alg, kpi_type, episode_count)
        split_idx_all_episodes.append(split_idx_per_episode)
        kpi_type = 'ue_energy_comm'
        time_steps, ue_energy_comm_episode = read_kpis_from_files(alg, kpi_type, episode_count)
        ue_energy_comm_all_episodes.append(ue_energy_comm_episode)

    # concatenate data of all episodes into single data structure
    df_compression = pd.DataFrame(compression_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    df_top1_accuracy = pd.DataFrame(top1_accuracy_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_split_idx = pd.DataFrame(split_idx_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    df_ue_energy_comm = pd.DataFrame(ue_energy_comm_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    return df_compression, df_top1_accuracy, df_split_idx, df_ue_energy_comm

def return_metrics_list_per_episode(episode_number, df_compression, df_top1_accuracy, df_split_idx, df_ue_energy_comm):
    x = df_compression.iloc[episode_number].to_list()
    y = df_top1_accuracy.iloc[episode_number].to_list()
    z = df_split_idx.iloc[episode_number].to_list()
    w = df_ue_energy_comm.iloc[episode_number].to_list()

    return x, y, z, w

def plot_compression_vs_accuracy(df_compression_list, df_top1_accuracy_list, df_split_idx_list, df_ue_energy_comm_list,
                                 n_episodes_to_train, total_episodes_train):
    # the methodology to follow is to first check which split config is being selected by the agent the most
    # check 26
    split_idx_to_plot = 81
    fig, ax = plt.subplots(layout='constrained')
    # data extraction for the different mechanisms
    # for optimum
    alg_idx = 0
    df_opt_compression = df_compression_list[alg_idx]
    df_opt_compression_list = []
    df_opt_top1_accuracy = df_top1_accuracy_list[alg_idx]
    df_opt_top1_accuracy_list = []
    df_opt_split_idx = df_split_idx_list[alg_idx]
    df_opt_split_idx_list = []
    df_opt_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_opt_ue_energy_comm_list = []

    # for rl/ddqn
    alg_idx = 1
    df_ddqn_compression = df_compression_list[alg_idx]
    df_ddqn_compression_list = []
    #print(df_ddqn_compression.iloc[0])
    df_ddqn_top1_accuracy = df_top1_accuracy_list[alg_idx]
    df_ddqn_top1_accuracy_list = []
    df_ddqn_split_idx = df_split_idx_list[alg_idx]
    df_ddqn_split_idx_list = []
    df_ddqn_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_ddqn_ue_energy_comm_list = []
    n_episodes_aft_train = total_episodes_train - n_episodes_to_train

    # for random
    alg_idx = 2
    df_random_compression = df_compression_list[alg_idx]
    df_random_compression_list = []
    df_random_top1_accuracy = df_top1_accuracy_list[alg_idx]
    df_random_top1_accuracy_list = []
    df_random_split_idx = df_split_idx_list[alg_idx]
    df_random_split_idx_list = []
    df_random_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_random_ue_energy_comm_list = []

    # for optimum
    for i in range(0, 9):
        x, y, z, w = return_metrics_list_per_episode(i, df_opt_compression, df_opt_top1_accuracy, df_opt_split_idx,
                                                     df_opt_ue_energy_comm)
        df_opt_compression_list.extend(x)
        df_opt_top1_accuracy_list.extend(y)
        df_opt_split_idx_list.extend(z)
        df_opt_ue_energy_comm_list.extend(w)
    # for ddqn
    for i in range(n_episodes_to_train-1, total_episodes_train):
        x, y, z, w = return_metrics_list_per_episode(i, df_ddqn_compression, df_ddqn_top1_accuracy, df_ddqn_split_idx,
                                                  df_ddqn_ue_energy_comm)
        df_ddqn_compression_list.extend(x)
        df_ddqn_top1_accuracy_list.extend(y)
        df_ddqn_split_idx_list.extend(z)
        df_ddqn_ue_energy_comm_list.extend(w)
    #print(x)
    # for random
    for i in range(0, 200):
        x, y, z, w = return_metrics_list_per_episode(i, df_random_compression, df_random_top1_accuracy, df_random_split_idx,
                                                     df_random_ue_energy_comm)
        df_random_compression_list.extend(x)
        df_random_top1_accuracy_list.extend(y)
        df_random_split_idx_list.extend(z)
        df_random_ue_energy_comm_list.extend(w)

    df_opt = pd.DataFrame({'compression': df_opt_compression_list, 'top1': df_opt_top1_accuracy_list,
                           'split_idx': df_opt_split_idx_list, 'ue_energy_comm': df_opt_ue_energy_comm_list})
    df_ddqn = pd.DataFrame({'compression': df_ddqn_compression_list, 'top1': df_ddqn_top1_accuracy_list,
                       'split_idx': df_ddqn_split_idx_list, 'ue_energy_comm': df_ddqn_ue_energy_comm_list})
    df_random = pd.DataFrame({'compression': df_random_compression_list, 'top1': df_random_top1_accuracy_list,
                              'split_idx': df_random_split_idx_list, 'ue_energy_comm': df_random_ue_energy_comm_list})
    new_df = pd.concat([df_opt, df_ddqn, df_random])
    mdf = pd.melt(new_df)
    ax = sns.boxplot(x='')
    # select rows from df that correspond to split_idx = split_idx_to_plot
    #df_subset = df.loc[df['split_idx'] == split_idx_to_plot]
    #df = pd.concat([x, y], axis=1)
    #print(df)
    #print(df_subset)
    #df_subset = df_subset.drop('split_idx', axis=1)
    df_ddqn = df_ddqn.drop('split_idx', axis=1)
    df_ddqn = df_ddqn.drop('top1', axis=1)
    #df_subset.boxplot(by='compression')
    df_ddqn.boxplot(by='compression')
    #df.rename(columns={1: 'compression', 1: 'top1'})
    #ax.scatter(x, y, marker='^', color='#072140', label='ddqn')
    #ax.grid()
    #ax.set_xlabel('ratio of original data to compressed data (rho)')
    #ax.set_ylabel('top1 accuracy confidence (%)')
    plt.show()


def main():
    """
    Main function that invokes other functions to read and parse kpis and plot them.

    Returns:

    """
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    algorithms = ['optimum', 'rl/ddqn', 'random']
    n_episodes = {'optimum': 9, 'rl/ddqn': 2000, 'random': 200}
    df_compression_list = []    # for each specified algorithm in 'algorithms'
    df_top1_accuracy_list = []  # for each specified algorithm in 'algorithms'
    df_split_idx_list = []      # for each specified algorithm in 'algorithms'
    df_ue_energy_comm_list = [] # for each specified algorithm in 'algorithms'

    n_episodes_to_train = 1000
    total_episodes_train = 2000

    for alg in algorithms:
        df_compression, df_top1_accuracy, df_split_idx, df_ue_energy_comm = parse_kpis(alg, n_episodes[alg])
        df_compression_list.append(df_compression)
        df_top1_accuracy_list.append(df_top1_accuracy)
        df_split_idx_list.append(df_split_idx)
        df_ue_energy_comm_list.append(df_ue_energy_comm)

    plot_compression_vs_accuracy(df_compression_list, df_top1_accuracy_list, df_split_idx_list, df_ue_energy_comm_list,
                                 n_episodes_to_train, total_episodes_train)

if __name__ == '__main__':
    main()