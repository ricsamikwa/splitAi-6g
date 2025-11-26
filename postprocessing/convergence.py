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
    ax1.grid()
    #ax1.legend()
    #ax2.legend()
    plt.savefig('results/{}/{}.png'.format(folder, outfile))
    plt.savefig('results/{}/{}.svg'.format(folder, outfile))
    plt.show()


def plot_metric(metric, df, outfile, folder):
    window = 10
    fig, ax = plt.subplots()
    roll_mean_loss = df[metric].rolling(window=window).mean()
    plt.plot(df['episode'], roll_mean_loss, label='ddqn')
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
        if metric == 'mean_loss':
            file = 'logs/{}/loss/loss_ep{}'.format(folder, ep_str)
            _, data = read_single_col_data(file, 'time', 'loss', float, float)
        else:
            file = 'logs/{}/reward/reward_ep{}'.format(folder, ep_str)
            _, data = read_single_col_data(file, 'time', 'reward', float, float)
        mean_data = statistics.mean(data)
        df.loc[idx] = pd.Series({'episode': ep, metric: mean_data})
        idx += 1
    return df



def main():
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    params = generate_scenario()
    # specify the number of episodes to be plotted
    n_episodes = 407
    order_to_convert = 1000
    folder = 'rl/a2c'
    #df_loss = generate_metric(params, n_episodes, order_to_convert, 'mean_loss')
    df_reward = generate_metric(folder, n_episodes, order_to_convert, 'mean_reward')

    # call this function to generate loss and reward charts separately
    #plot_metric('mean_loss', df_loss, 'mean_loss', folder)
    plot_metric('mean_reward', df_reward, 'mean_reward', folder)

    # call this function to plot both loss and reward in the same chart
    #plot_metrics_together('mean_loss', 'mean_reward', df_loss, df_reward, 'loss_reward', folder)


if __name__ == '__main__':
    main()
