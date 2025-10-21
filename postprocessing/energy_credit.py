import os
import pandas
import matplotlib.pyplot as plt

from utils.logging_utils import return_order, parse_episode_number

def plot_kpis_in_episode(ue_energy_comp_per_algorithm, ue_energy_comm_per_algorithm, flops_offloaded_per_algorithm,
                         y_net_per_algorithm, algorithms):
    """

    Args:
        inference_time_data:
        energy_credit_data:
        algorithm:

    Returns:

    """
    # instantiate first axis
    fig, ax1 = plt.subplots()
    window = 10
    ax1.set_xlabel('Simulation time in s')
    ax1.set_ylabel('UE energy comp (J)')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accumulated energy credits used')
    for alg in algorithms:
        ue_energy_comp_data = ue_energy_comp_per_algorithm[alg]
        ue_energy_comm_data = ue_energy_comm_per_algorithm[alg]
        flops_offloaded_data = flops_offloaded_per_algorithm[alg]
        y_net_data = y_net_per_algorithm[alg]
        print(flops_offloaded_data)
        print(y_net_data)
        y = flops_offloaded_data['flops_off'] + y_net_data['y_net']
        #y = flops_offloaded_data['flops_off']
        #y = y_net_data['y_net']
        ax1.plot(ue_energy_comp_data['time_step'], ue_energy_comp_data['ue_energy_comp'], color='#072140', marker='o',
                 label='ue energy comp')
        #ax1.plot(ue_energy_comm_data['time_step'], ue_energy_comm_data['ue_energy_comm'], color='r', marker='o',
        #         label='{}'.format(alg))
        ax2.plot(flops_offloaded_data['time_step'], y, linestyle='dotted', color='#9ABCE4', marker='D',
                 label='energy credits')

    ax1.grid()
    ax1.legend(loc='upper left')
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
    path = 'logs/{}/system/{}.csv'.format(algorithm, filename)

    kpi_data = pandas.read_csv(path)
    return kpi_data

def main():
    """
    Plots inference latency and energy credit usage in a specified episode.
    Returns:

    """
    path_parent = os.path.dirname(os.getcwd())
    os.chdir(path_parent)

    algorithm = ['rl/ddqn']
    episode_to_plot = {'optimum': 1, 'rl/ddqn': 2, 'random': 1, 'fixed': 1}
    order_to_convert = {'optimum': 1, 'rl/ddqn': 1000, 'random': 1, 'fixed': 1}
    flops_offloaded_per_algorithm = {}
    inference_time_per_algorithm = {}
    ue_energy_comp_per_algorithm = {}
    ue_energy_comm_per_algorithm = {}
    y_net_per_algorithm ={}
    for alg in algorithm:
        #energy_credit_data = read_kpis('energy_credit', alg, episode_to_plot[alg], order_to_convert[alg])
        flops_offloaded_data = read_kpis('flops_off', alg, episode_to_plot[alg], order_to_convert[alg])
        ue_energy_comp_data = read_kpis('ue_energy_comp', alg, episode_to_plot[alg], order_to_convert[alg])
        ue_energy_comm_data = read_kpis('ue_energy_comm', alg, episode_to_plot[alg], order_to_convert[alg])
        y_net_data = read_kpis('y_net', alg, episode_to_plot[alg], order_to_convert[alg])
        #inference_time_data = read_kpis('inference_time', alg, episode_to_plot[alg], order_to_convert[alg])
        #energy_credit_per_algorithm[alg] = energy_credit_data
        #inference_time_per_algorithm[alg] = inference_time_data
        flops_offloaded_per_algorithm[alg] = flops_offloaded_data
        ue_energy_comp_per_algorithm[alg] = ue_energy_comp_data
        ue_energy_comm_per_algorithm[alg] = ue_energy_comm_data
        y_net_per_algorithm[alg] =y_net_data
    plot_kpis_in_episode(ue_energy_comp_per_algorithm, ue_energy_comm_per_algorithm, flops_offloaded_per_algorithm, y_net_per_algorithm, algorithm)


if __name__ == '__main__':
    main()