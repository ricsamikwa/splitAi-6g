import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics


from utils.scenario_generator import generate_scenario
from utils.logging_utils import read_single_col_data, return_order, parse_episode_number

def plot_metrics_together(metric1, metric2, df1, df2, outfile, folder):
    window = 10
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('episode')
    ax1.set_ylabel('{}'.format(metric1))
    roll_mean_metric1 = df1[metric1].rolling(window=window).mean()
    ax1.plot(df1['episode'], roll_mean_metric1, color='#072140', label=metric1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('{}'.format(metric2))
    roll_mean_metric2 = df2[metric2].rolling(window=window).mean()
    ax2.plot(df2['episode'], roll_mean_metric2, color='#9ABCE4', label=metric2)
    #ax2.yscale('log')
    ax1.grid()
    #ax1.legend()
    #ax2.legend()
    plt.savefig('results/{}/{}.png'.format(folder, outfile))
    plt.savefig('results/{}/{}.svg'.format(folder, outfile))
    plt.show()


def plot_metric(metric, df, outfile, folder):
    window = 50
    fig, ax = plt.subplots()
    roll_mean_metric = df[metric].rolling(window=window).mean()
    plt.plot(df['episode'], roll_mean_metric, label='{}'.format(folder))
    ax.set_xlabel('episode')
    ax.set_ylabel('{}, window {}'.format(metric, window))
    ax.legend()
    ax.grid()
    plt.savefig('results/{}/{}.png'.format(folder, outfile))
    plt.savefig('results/{}/{}.svg'.format(folder, outfile))
    plt.show()

def generate_metric(folder, n_episodes, order_to_convert, metric):
    order = return_order(order_to_convert)
    df = pd.DataFrame(columns=['episode', metric])
    idx = 0
    for ep in range(1, n_episodes + 1):
        ep_str = parse_episode_number(order, ep)
        if metric == 'loss' or metric == 'actor_loss' or metric == 'critic_loss':
            file = 'logs/{}/loss/{}_ep{}'.format(folder, metric, ep_str)
            _, data = read_single_col_data(file, 'time', 'loss', float, float)
        elif metric == 'reward':
            file = 'logs/{}/reward/reward_ep{}'.format(folder, ep_str)
            _, data = read_single_col_data(file, 'time', 'reward', float, float)
        elif metric == 'advantage':
            file = 'logs/{}/advantage/advantage_ep{}'.format(folder, ep_str)
            _, data = read_single_col_data(file, 'time', 'advantage', float, float)
        else:
            file = 'logs/{}/entropy/entropy_ep{}'.format(folder, ep_str)
            _, data = read_single_col_data(file, 'time', 'entropy', float, float)
        mean_data = statistics.mean(data)
        df.loc[idx] = pd.Series({'episode': ep, metric: mean_data})
        idx += 1
    return df



def main():
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    params = generate_scenario()
    # specify the number of episodes to be plotted
    n_episodes = 1097
    order_to_convert = 1000

    # ---- for DDQN -----
    folder = 'rl/ddqn'
    df_loss = generate_metric(folder, n_episodes, order_to_convert, 'loss')
    df_reward = generate_metric(folder, n_episodes, order_to_convert, 'reward')

    # ---- for A2C -----
    #folder = 'rl/a2c'
    #df_actor_loss = generate_metric(folder, n_episodes, order_to_convert, 'actor_loss')
    #df_critic_loss = generate_metric(folder, n_episodes, order_to_convert, 'critic_loss')
    #df_reward = generate_metric(folder, n_episodes, order_to_convert, 'reward')
    #df_advantage = generate_metric(folder, n_episodes, order_to_convert, 'advantage')
    #df_entropy = generate_metric(folder, n_episodes, order_to_convert, 'entropy')

    # ---- for DDQN -----
    plot_metric('loss', df_loss, 'loss', folder)
    plot_metric('reward', df_reward, 'reward', folder)

    # call this function to generate loss and reward charts separately for A2C
    #plot_metric('actor_loss', df_actor_loss, 'actor_loss', folder)
    #plot_metric('critic_loss', df_critic_loss, 'critic_loss', folder)
    #plot_metric('reward', df_reward, 'reward', folder)
    #plot_metric('advantage', df_advantage, 'advantage', folder)
    #plot_metric('entropy', df_entropy, 'entropy', folder)

    # call this function to plot both loss and reward in the same chart
    #plot_metrics_together('mean_loss', 'mean_reward', df_loss, df_reward, 'loss_reward', folder)


if __name__ == '__main__':
    main()
