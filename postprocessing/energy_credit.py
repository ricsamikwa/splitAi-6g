import os
import pandas
import matplotlib.pyplot as plt
from utils.logging_utils import return_order, parse_episode_number

def plot_kpis_in_episode(inference_time_per_algorithm, energy_credit_per_algorithm, algorithms):
    """

    Args:
        inference_time_data:
        energy_credit_data:
        algorithm:

    Returns:

    """
    # instantiate first axis
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Simulation time in s')
    ax1.set_ylabel('Inference Latency')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Energy credit usage')
    for alg in algorithms:
        inference_time_data = inference_time_per_algorithm[alg]
        energy_credit_data = energy_credit_per_algorithm[alg]
        ax1.plot(inference_time_data['time_step'], inference_time_data['inference_time'], marker='o', label='{}'.format(alg))
        ax2.plot(energy_credit_data['time_step'], energy_credit_data['energy_credit'], linestyle='dotted', marker='D',
                 label='{}'.format(alg))

    ax1.grid()
    #ax1.legend(loc='upper left')
    ax2.legend(loc='lower left')
    plt.savefig('results/energy_credit_inference_time.png')
    plt.savefig('results/energy_credit_inference_time.svg')
    plt.show()


def read_kpis(kpi, algorithm, episode_to_plot, order_to_convert):
    """
    Script that loads the specified kpi (inference time or energy credit) to a dataframe.
    Args:
        kpi (str): the kpi to load i.e. inference time or energy credit.
        algorithm (str): the algorithm whose kpi is to be read.
        episode_to_plot (int): the episode for which the plot is to be generated.
        order_to_convert (int): the order of the algorithm whose kpi data is to be read.

    Returns:
        A pandas dataframe with the specified kpi data.
    """
    order = return_order(order_to_convert)
    episode_count = parse_episode_number(order, episode_to_plot)

    filename = '{}_{}'.format(kpi, episode_count)
    path = 'logs/comparison/{}/{}.csv'.format(algorithm, filename)

    kpi_data = pandas.read_csv(path)
    return kpi_data

def main():
    """
    Plots inference latency and energy credit usage in a specified episode.
    Returns:

    """
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    algorithm = ['optimum', 'rl/ddqn']
    episode_to_plot = {'optimum': 1, 'rl/ddqn': 1000, 'random': 1, 'fixed': 1}
    order_to_convert = {'optimum': 1, 'rl/ddqn': 1000, 'random': 1, 'fixed': 1}
    energy_credit_per_algorithm = {}
    inference_time_per_algorithm = {}
    for alg in algorithm:
        energy_credit_data = read_kpis('energy_credit', alg, episode_to_plot[alg], order_to_convert[alg])
        inference_time_data = read_kpis('inference_time', alg, episode_to_plot[alg], order_to_convert[alg])
        energy_credit_per_algorithm[alg] = energy_credit_data
        inference_time_per_algorithm[alg] = inference_time_data
    plot_kpis_in_episode(inference_time_per_algorithm, energy_credit_per_algorithm, algorithm)


if __name__ == '__main__':
    main()