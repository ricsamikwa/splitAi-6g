import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from utils.logging_utils import return_order, parse_episode_number


def read_kpis_from_files(folder, kpi_type, episode_count):
    if kpi_type == 'top1':
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

    # concatenate data of all episodes into single data structure
    df_compression = pd.DataFrame(compression_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    df_top1_accuracy = pd.DataFrame(top1_accuracy_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_split_idx = pd.DataFrame(split_idx_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    return df_compression, df_top1_accuracy, df_split_idx

def return_metrics_list_per_episode(episode_number, df_compression, df_top1_accuracy, df_split_idx):
    x = df_compression.iloc[episode_number].to_list()
    y = df_top1_accuracy.iloc[episode_number].to_list()
    z = df_split_idx.iloc[episode_number].to_list()

    return x, y, z

def plot_compression_vs_accuracy(df_compression_list, df_top1_accuracy_list, df_split_idx_list, n_episodes_to_train,
                                 total_episodes_train):
    # the methodology to follow is to first check which split config is being selected by the agent the most
    # check 26
    split_idx_to_plot = 29
    fig, ax = plt.subplots(layout='constrained')
    # plot only for rl/ddqn now
    df_ddqn_compression = df_compression_list[0]
    df_ddqn_compression_list = []
    #print(df_ddqn_compression.iloc[0])
    df_ddqn_top1_accuracy = df_top1_accuracy_list[0]
    df_ddqn_top1_accuracy_list = []
    df_ddqn_split_idx = df_split_idx_list[0]
    df_ddqn_split_idx_list = []
    n_episodes_aft_train = total_episodes_train - n_episodes_to_train

    #df_ddqn_compression['mean'] = df_ddqn_compression.mean(axis=1)
    #mean_compression_ddqn = df_ddqn_compression['mean'][n_episodes_aft_train:total_episodes_train].mean()
    #df_ddqn_top1_accuracy['mean'] = df_ddqn_top1_accuracy.mean(axis=1)
    #mean_top1_ddqn = df_ddqn_top1_accuracy['mean'][n_episodes_aft_train:total_episodes_train].mean()

    for i in range(n_episodes_to_train-1, total_episodes_train):
        x, y, z = return_metrics_list_per_episode(i, df_ddqn_compression, df_ddqn_top1_accuracy, df_ddqn_split_idx)
        df_ddqn_compression_list.extend(x)
        df_ddqn_top1_accuracy_list.extend(y)
        df_ddqn_split_idx_list.extend(z)
    #print(x)
    df = pd.DataFrame({'compression': df_ddqn_compression_list, 'top1': df_ddqn_top1_accuracy_list,
                       'split_idx': df_ddqn_split_idx_list})
    # select rows from df that correspond to split_idx = split_idx_to_plot
    df_subset = df.loc[df['split_idx'] == split_idx_to_plot]
    #df = pd.concat([x, y], axis=1)
    print(df)
    print(df_subset)
    df_subset = df_subset.drop('split_idx', axis=1)
    df_subset.boxplot(by='compression')
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

    algorithms = ['rl/ddqn']
    n_episodes = {'rl/ddqn': 100}
    df_compression_list = []    # for each specified algorithm in 'algorithms'
    df_top1_accuracy_list = []  # for each specified algorithm in 'algorithms'
    df_split_idx_list = []      # for each specified algorithm in 'algorithms'

    n_episodes_to_train = 60
    total_episodes_train = 100

    for alg in algorithms:
        df_compression, df_top1_accuracy, df_split_idx = parse_kpis(alg, n_episodes[alg])
        df_compression_list.append(df_compression)
        df_top1_accuracy_list.append(df_top1_accuracy)
        df_split_idx_list.append(df_split_idx)

    plot_compression_vs_accuracy(df_compression_list, df_top1_accuracy_list, df_split_idx_list, n_episodes_to_train, total_episodes_train)

if __name__ == '__main__':
    main()