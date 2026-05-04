import statistics
from utils.logging_utils import return_order, parse_episode_number
import csv, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_kpis_from_files(folder, kpi_type, weight, episode_count):
    file = 'logs/{}/comparison/omega1_{}/splits/{}_{}.csv'.format(folder, weight, kpi_type, episode_count)
    data_timestep = []
    data_kpi = []
    with open(file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for k, item in enumerate(reader):
            if k != 0:
                data_timestep.append(int(item[0]))
                data_kpi.append(str(item[1]))
    return data_timestep, data_kpi


def parse_kpis(alg, n_episodes, weight):
    time_steps = []
    splits_all_episodes = []
    order = return_order(n_episodes=1000)
    for episode in range(1, n_episodes + 1):
        episode_count = parse_episode_number(order, episode)
        kpi_type = 'split'
        time_steps, splits_per_episode = read_kpis_from_files(alg, kpi_type, weight, episode_count)
        splits_all_episodes.append(splits_per_episode)
    df_split_idx = pd.DataFrame(splits_all_episodes, columns=time_steps,
                                index=[ep for ep in range(1, n_episodes + 1)])
    return df_split_idx


def extractSplits(s):
    # apply transformation to the string
    n_commas = 0
    close_bracket = False
    ue = ''
    for char in s:
        if char == ',':
            n_commas += 1
        if char == ')':
            break
        if n_commas > 2:
            break
        if n_commas == 2 and not close_bracket and char.isalnum():
            ue += char
    return int(ue)

def plot_splits(df_splits_list):
    fig, ax1 = plt.subplots(layout='constrained')
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    start_episode = 4500
    end_episode = 5000
    ue_computations_all = []
    for i in range(len(weights)):
        df_splits_weight = df_splits_list[i]    # for each weight
        ue_computation_means_per_episode = []
        for episode in range(start_episode, end_episode + 1):
            ue_computations = []
            for j in range(1, 51):
                ue = extractSplits(df_splits_weight[j][episode])
                ue_percentage = (ue/18) * 100
                ue_computations.append(ue_percentage)
            ue_computation_means_per_episode.append(statistics.mean(ue_computations))
        ue_computations_all.append(statistics.mean(ue_computation_means_per_episode))
    print(ue_computations_all)
    r = np.arange(len(weights))  # the label locations
    plt.bar(r, ue_computations_all)
    ax1.set_xlabel('weight')
    ax1.set_ylabel('percentage of computations on ue')
    ax1.set_xticks(r)
    ax1.set_xticklabels(weights)
    plt.grid()
    plt.show()



def main():
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    df_splits_list = []  # for each weight in weights

    for weight in weights:
        df_splits = parse_kpis('rl/ddqn', 5000, weight)
        #print(df_splits[1])
        df_splits_list.append(df_splits)
    test = '[(0, 0, 10), (1, 10, 14), (2, 14, 18), (3, 18, 18)]'
    test = df_splits_list[0][1][4500]  # weight, time_stamp, episode
    #print(test)
    #print(extractSplits(test))
    plot_splits(df_splits_list)

if __name__ == '__main__':
    main()
