import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from utils.logging_utils import return_order, parse_episode_number


def read_kpis_from_files(folder, kpi_type, episode_count):
    if kpi_type == 'top1' or kpi_type == 'ue_energy_comm' or kpi_type == 'ue_energy_comp':
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
    if alg != 'rl/ddqn':
        order = return_order(n_episodes)
    else:
        order = return_order(n_episodes=1000)
    time_steps = []
    compression_all_episodes = []
    top1_accuracy_all_episodes = []
    split_idx_all_episodes = []
    ue_energy_comm_all_episodes = []
    ue_energy_comp_all_episodes = []
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
        kpi_type = 'ue_energy_comp'
        time_steps, ue_energy_comp_per_episode = read_kpis_from_files(alg, kpi_type, episode_count)
        ue_energy_comp_all_episodes.append(ue_energy_comp_per_episode)

    # concatenate data of all episodes into single data structure
    df_compression = pd.DataFrame(compression_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    df_top1_accuracy = pd.DataFrame(top1_accuracy_all_episodes, columns=time_steps,
                                     index=[ep for ep in range(1, n_episodes + 1)])
    df_split_idx = pd.DataFrame(split_idx_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    df_ue_energy_comm = pd.DataFrame(ue_energy_comm_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    df_ue_energy_comp = pd.DataFrame(ue_energy_comp_all_episodes, columns=time_steps,
                                  index=[ep for ep in range(1, n_episodes + 1)])
    return df_compression, df_top1_accuracy, df_split_idx, df_ue_energy_comm, df_ue_energy_comp

def return_metrics_list_per_episode(episode_number, df_compression, df_top1_accuracy, df_split_idx, df_ue_energy_comm,
                                    df_ue_energy_comp):
    x = df_compression.iloc[episode_number].to_list()
    y = df_top1_accuracy.iloc[episode_number].to_list()
    z = df_split_idx.iloc[episode_number].to_list()
    w = df_ue_energy_comm.iloc[episode_number].to_list()
    k = df_ue_energy_comp.iloc[episode_number].to_list()

    return x, y, z, w, k

def plot_compression_vs_accuracy(df_compression_list, df_top1_accuracy_list, df_split_idx_list, df_ue_energy_comm_list,
                                 df_ue_energy_comp_list, n_episodes_to_train, total_episodes_train):
    # the methodology to follow is to first check which split config is being selected by the agent the most
    # check 26
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
    df_opt_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_opt_ue_energy_comp_list = []

    # for rl/ddqn
    alg_idx = 1
    df_ddqn_compression = df_compression_list[alg_idx]
    df_ddqn_compression_list = []
    df_ddqn_top1_accuracy = df_top1_accuracy_list[alg_idx]
    df_ddqn_top1_accuracy_list = []
    df_ddqn_split_idx = df_split_idx_list[alg_idx]
    df_ddqn_split_idx_list = []
    df_ddqn_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_ddqn_ue_energy_comm_list = []
    df_ddqn_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_ddqn_ue_energy_comp_list = []

    # for rl/a2c
    alg_idx = 2
    df_a2c_compression = df_compression_list[alg_idx]
    df_a2c_compression_list = []
    df_a2c_top1_accuracy = df_top1_accuracy_list[alg_idx]
    df_a2c_top1_accuracy_list = []
    df_a2c_split_idx = df_split_idx_list[alg_idx]
    df_a2c_split_idx_list = []
    df_a2c_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_a2c_ue_energy_comm_list = []
    df_a2c_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_a2c_ue_energy_comp_list = []

    # for random
    alg_idx = 3
    df_random_compression = df_compression_list[alg_idx]
    df_random_compression_list = []
    df_random_top1_accuracy = df_top1_accuracy_list[alg_idx]
    df_random_top1_accuracy_list = []
    df_random_split_idx = df_split_idx_list[alg_idx]
    df_random_split_idx_list = []
    df_random_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_random_ue_energy_comm_list = []
    df_random_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_random_ue_energy_comp_list = []

    # for heuristic
    alg_idx = 4
    df_hr_compression = df_compression_list[alg_idx]
    df_hr_compression_list = []
    df_hr_top1_accuracy = df_top1_accuracy_list[alg_idx]
    df_hr_top1_accuracy_list = []
    df_hr_split_idx = df_split_idx_list[alg_idx]
    df_hr_split_idx_list = []
    df_hr_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_hr_ue_energy_comm_list = []
    df_hr_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_hr_ue_energy_comp_list = []

    # for fixed
    alg_idx = 4
    df_fixed_compression = df_compression_list[alg_idx]
    df_fixed_compression_list = []
    df_fixed_top1_accuracy = df_top1_accuracy_list[alg_idx]
    df_fixed_top1_accuracy_list = []
    df_fixed_split_idx = df_split_idx_list[alg_idx]
    df_fixed_split_idx_list = []
    df_fixed_ue_energy_comm = df_ue_energy_comm_list[alg_idx]
    df_fixed_ue_energy_comm_list = []
    df_fixed_ue_energy_comp = df_ue_energy_comp_list[alg_idx]
    df_fixed_ue_energy_comp_list = []

    # for optimum
    for i in range(0, 1):
        x, y, z, w, k = return_metrics_list_per_episode(i, df_opt_compression, df_opt_top1_accuracy, df_opt_split_idx,
                                                     df_opt_ue_energy_comm, df_opt_ue_energy_comp)
        df_opt_compression_list.extend(x)
        df_opt_top1_accuracy_list.extend(y)
        df_opt_split_idx_list.extend(z)
        df_opt_ue_energy_comm_list.extend(w)
        df_opt_ue_energy_comp_list.extend(k)
    # for ddqn
    for i in range(n_episodes_to_train['rl/ddqn']-1, total_episodes_train['rl/ddqn']):
        x, y, z, w, k = return_metrics_list_per_episode(i, df_ddqn_compression, df_ddqn_top1_accuracy, df_ddqn_split_idx,
                                                  df_ddqn_ue_energy_comm, df_ddqn_ue_energy_comp)
        df_ddqn_compression_list.extend(x)
        df_ddqn_top1_accuracy_list.extend(y)
        df_ddqn_split_idx_list.extend(z)
        df_ddqn_ue_energy_comm_list.extend(w)
        df_ddqn_ue_energy_comp_list.extend(k)
    # for a2c
    for i in range(n_episodes_to_train['rl/a2c'] - 1, total_episodes_train['rl/a2c']):
        x, y, z, w, k = return_metrics_list_per_episode(i, df_a2c_compression, df_a2c_top1_accuracy, df_a2c_split_idx,
                                                     df_a2c_ue_energy_comm, df_a2c_ue_energy_comp)
        df_a2c_compression_list.extend(x)
        df_a2c_top1_accuracy_list.extend(y)
        df_a2c_split_idx_list.extend(z)
        df_a2c_ue_energy_comm_list.extend(w)
        df_a2c_ue_energy_comp_list.extend(k)
    # for random
    for i in range(0, 200):
        x, y, z, w, k = return_metrics_list_per_episode(i, df_random_compression, df_random_top1_accuracy, df_random_split_idx,
                                                     df_random_ue_energy_comm, df_random_ue_energy_comp)
        df_random_compression_list.extend(x)
        df_random_top1_accuracy_list.extend(y)
        df_random_split_idx_list.extend(z)
        df_random_ue_energy_comm_list.extend(w)
        df_random_ue_energy_comp_list.extend(k)

    # # for heuristic
    # for i in range(0, 100):
    #     x, y, z, w, k = return_metrics_list_per_episode(i, df_hr_compression, df_hr_top1_accuracy, df_hr_split_idx,
    #                                                     df_hr_ue_energy_comm, df_hr_ue_energy_comp)
    #     df_hr_compression_list.extend(x)
    #     df_hr_top1_accuracy_list.extend(y)
    #     df_hr_split_idx_list.extend(z)
    #     df_hr_ue_energy_comm_list.extend(w)
    #     df_hr_ue_energy_comp_list.extend(k)
    # for fixed
    for i in range(0, 9):
        x, y, z, w, k = return_metrics_list_per_episode(i, df_hr_compression, df_hr_top1_accuracy, df_hr_split_idx,
                                                        df_hr_ue_energy_comm, df_hr_ue_energy_comp)
        df_fixed_compression_list.extend(x)
        df_fixed_top1_accuracy_list.extend(y)
        df_fixed_split_idx_list.extend(z)
        df_fixed_ue_energy_comm_list.extend(w)
        df_fixed_ue_energy_comp_list.extend(k)

    df_opt = pd.DataFrame({'compression': df_opt_compression_list, 'top1': df_opt_top1_accuracy_list,
                           'split_idx': df_opt_split_idx_list, 'ue_energy_comm': df_opt_ue_energy_comm_list,
                           'ue_energy_comp': df_opt_ue_energy_comp_list})
    df_ddqn = pd.DataFrame({'compression': df_ddqn_compression_list, 'top1': df_ddqn_top1_accuracy_list,
                       'split_idx': df_ddqn_split_idx_list, 'ue_energy_comm': df_ddqn_ue_energy_comm_list,
                            'ue_energy_comp': df_ddqn_ue_energy_comp_list})
    df_a2c = pd.DataFrame({'compression': df_a2c_compression_list, 'top1': df_a2c_top1_accuracy_list,
                           'split_idx': df_a2c_split_idx_list, 'ue_energy_comm': df_a2c_ue_energy_comm_list,
                           'ue_energy_comp': df_a2c_ue_energy_comp_list})
    df_random = pd.DataFrame({'compression': df_random_compression_list, 'top1': df_random_top1_accuracy_list,
                              'split_idx': df_random_split_idx_list, 'ue_energy_comm': df_random_ue_energy_comm_list,
                              'ue_energy_comp': df_random_ue_energy_comp_list})
    # df_heuristic = pd.DataFrame({'compression': df_hr_compression_list, 'top1': df_hr_top1_accuracy_list,
    #                           'split_idx': df_hr_split_idx_list, 'ue_energy_comm': df_hr_ue_energy_comm_list,
    #                           'ue_energy_comp': df_hr_ue_energy_comp_list})
    df_fixed = pd.DataFrame({'compression': df_fixed_compression_list, 'top1': df_fixed_top1_accuracy_list,
                              'split_idx': df_fixed_split_idx_list, 'ue_energy_comm': df_fixed_ue_energy_comm_list,
                              'ue_energy_comp': df_fixed_ue_energy_comp_list})

    df_opt['algorithm'] = 'opt'
    df_ddqn['algorithm'] = 'ddqn'
    df_a2c['algorithm'] = 'a2c'
    df_random['algorithm'] = 'random'
    #df_heuristic['algorithm'] = 'heuristic'
    df_fixed['algorithm'] = 'fixed'
    df_all = pd.concat([df_opt, df_ddqn, df_a2c, df_random], ignore_index=True)
    palette = {'opt': '#072140', 'ddqn': '#114584', 'a2c': '#165DB1', 'random': '#9ABCE4', 'fixed': '#8F81EA'}
    palette = {'opt': '#072140', 'ddqn': '#165DB1', 'a2c': '#9ABCE4', 'random': '#8F81EA'}
    #df_all.boxplot(column='ue_energy_comm', by=['compression', 'algorithm'])
    # sns.boxplot(
    #     data=df_all,
    #     x="compression",
    #     y="ue_energy_comm",
    #     hue="algorithm"
    # )
    # sns.violinplot(
    #     data=df_all,
    #     x="compression",
    #     y="ue_energy_comm",
    #     hue="algorithm",
    #     palette=palette
    # )
    # ax = sns.boxenplot(
    #     data=df_all,
    #     x="compression",
    #     y="ue_energy_comm",
    #     hue="algorithm",
    #     palette=palette,
    #     width=0.7,
    #     k_depth="proportion"
    # )
    # sns.stripplot(
    #     data=df_all,
    #     x="compression",
    #     y="ue_energy_comm",
    #     hue="algorithm",
    #     palette=palette,
    #     dodge=True,
    #     alpha=0.35,
    #     size=3
    # )
    # ----------------- for journal ----------------
    sns.barplot(
        data=df_all,
        x="compression",
        y="ue_energy_comm",
        hue="algorithm",
        estimator="mean",
        errorbar=("ci", 95),
        palette=palette
    )
    # for top1 accuracy confidence
    # sns.barplot(
    #     data=df_all,
    #     x="compression",
    #     y="top1",
    #     hue="algorithm",
    #     estimator="mean",
    #     errorbar=("ci", 95),
    #     palette=palette
    # )
    # for ue_energy_comp
    # sns.barplot(
    #     data=df_all,
    #     x="compression",
    #     y="ue_energy_comp",
    #     hue="algorithm",
    #     estimator="mean",
    #     errorbar=("ci", 95),
    #     palette=palette
    # )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:4], labels[:4], title="Algorithm")
    #plt.xticks(rotation=45)
    #plt.suptitle('')
    plt.grid()
    plt.savefig('results/journal/comm_vs_rho.png')
    plt.savefig('results/journal/comm_vs_rho.svg')
    plt.show()


def main():
    """
    Main function that invokes other functions to read and parse kpis and plot them.

    Returns:

    """
    # change the parent path to run this script independently
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    #algorithms = ['optimum', 'rl/ddqn', 'random']
    algorithms = ['optimum', 'rl/ddqn', 'rl/a2c', 'random', 'fixed']
    #n_episodes = {'optimum': 9, 'rl/ddqn': 2000, 'random': 200}
    n_episodes = {'optimum': 1, 'rl/ddqn': 5000, 'rl/a2c': 5000, 'random': 200, 'heuristic': 100, 'fixed': 9}
    df_compression_list = []    # for each specified algorithm in 'algorithms'
    df_top1_accuracy_list = []  # for each specified algorithm in 'algorithms'
    df_split_idx_list = []      # for each specified algorithm in 'algorithms'
    df_ue_energy_comm_list = [] # for each specified algorithm in 'algorithms'
    df_ue_energy_comp_list = []  # for each specified algorithm in 'algorithms'

    n_episodes_to_train = {'rl/ddqn': 1500, 'rl/a2c': 3000}
    total_episodes_train = {'rl/ddqn': 5000, 'rl/a2c': 5000}

    for alg in algorithms:
        df_compression, df_top1_accuracy, df_split_idx, df_ue_energy_comm, df_ue_energy_comp = parse_kpis(alg, n_episodes[alg])
        df_compression_list.append(df_compression)
        df_top1_accuracy_list.append(df_top1_accuracy)
        df_split_idx_list.append(df_split_idx)
        df_ue_energy_comm_list.append(df_ue_energy_comm)
        df_ue_energy_comp_list.append(df_ue_energy_comp)

    plot_compression_vs_accuracy(df_compression_list, df_top1_accuracy_list, df_split_idx_list, df_ue_energy_comm_list,
                                 df_ue_energy_comp_list, n_episodes_to_train, total_episodes_train)

if __name__ == '__main__':
    main()