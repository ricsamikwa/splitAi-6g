import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics


from utils.scenario_generator import generate_scenario
from utils.logging_utils import read_single_col_data, return_order, parse_episode_number


def plot_metric(metric, df, outfile, folder):
    window = 10
    fig, ax = plt.subplots()
    roll_mean_loss = df[metric].rolling(window=window).mean()
    plt.plot(df['episode'], roll_mean_loss, marker='o', label='ddqn')
    ax.set_xlabel('episode')
    ax.set_ylabel('{}, window {}'.format(metric, window))
    ax.legend()
    ax.grid()
    plt.savefig('results/{}/{}.png'.format(folder, outfile))
    plt.show()

def generate_metric(params, n_episodes, order_to_convert, metric):
    folder = 'rl/ddqn'
    order = return_order(order_to_convert)
    df = pd.DataFrame(columns=['episode', metric])
    idx = 0
    for ep in range(1, n_episodes + 1):
        ep_str = parse_episode_number(order, ep)
        if metric == 'mean_loss':
            file = 'logs/rl/ddqn/loss/loss_ep{}'.format(ep_str)
            _, data = read_single_col_data(file, 'time', 'loss', float, float)
        else:
            file = 'logs/rl/ddqn/reward/reward_ep{}'.format(ep_str)
            _, data = read_single_col_data(file, 'time', 'reward', float, float)
        mean_data = statistics.mean(data)
        df.loc[idx] = pd.Series({'episode': ep, metric: mean_data})
        idx += 1
    outfile = metric
    plot_metric(metric, df, outfile, folder)


def main():
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    params = generate_scenario()
    # specify the number of episodes to be plotted
    n_episodes = 50
    order_to_convert = 100
    generate_metric(params, n_episodes, order_to_convert, 'mean_loss')
    generate_metric(params, n_episodes, order_to_convert, 'mean_reward')


if __name__ == '__main__':
    main()
