"""
scenario_generator.py

Utility function for generating all config parameters specific to the scenario

"""
import configparser
import json


def generate_scenario():
    """
    Generates the scenario params.
    Returns:
        All read config parameters enclosed in a dict.
    """
    scenario_params = pack_parameters('config.ini')
    return scenario_params


def pack_parameters(filename):
    """
    Function to read config params from file using ConfigParser.
    Args:
        filename (str): the name of the config file to read from.

    Returns:
        All read config parameters enclosed in a dict.
    """
    config = configparser.ConfigParser()
    config.read(filename)

    # read all scenario parameters from config file
    weight_inference_time = float(config['ALGORITHM']['WEIGHT_INFERENCE_TIME'])
    max_energy_credit = int(config['ALGORITHM']['MAX_ENERGY_CREDIT'])
    max_inference_latency = float(config['ALGORITHM']['MAX_INFERENCE_LATENCY'])
    split_algorithm = int(config['ALGORITHM']['SPLIT_ALGORITHM'])
    rl_algorithm = int(config['ALGORITHM']['RL_ALGORITHM'])
    inference = int(config['ALGORITHM']['INFERENCE'])
    n_episodes = int(config['ALGORITHM']['N_EPISODES'])
    episode_duration = int(config['ALGORITHM']['EPISODE_DURATION'])
    time_interval = int(config['ALGORITHM']['TIME_INTERVAL'])
    start_episode = int(config['ALGORITHM']['START_EPISODE'])
    compression_rates = json.loads(config['ALGORITHM']['COMPRESSION_RATES'])
    accuracy_decrease = int(config['ALGORITHM']['ACCURACY_DECREASE'])
    weight_accuracy = float(config['ALGORITHM']['WEIGHT_ACCURACY'])
    top1_flag = int(config['ALGORITHM']['TOP1_FLAG'])
    top1_bins = json.loads(config['ALGORITHM']['TOP1_BINS'])
    param_path = config['ALGORITHM']['PARAM_PATH']

    n_hidden_layer = int(config['DRL_HYPERPARAMETERS']['N_HIDDEN_LAYER'])
    batch_size = int(config['DRL_HYPERPARAMETERS']['BATCH_SIZE'])
    buffer_size = int(config['DRL_HYPERPARAMETERS']['BUFFER_SIZE'])
    target_update = int(config['DRL_HYPERPARAMETERS']['TARGET_UPDATE'])
    target_update_policy = int(config['DRL_HYPERPARAMETERS']['TARGET_UPDATE_POLICY'])
    lr = float(config['DRL_HYPERPARAMETERS']['LR'])
    discount_factor = float(config['DRL_HYPERPARAMETERS']['DISCOUNT_FACTOR'])
    tau = float(config['DRL_HYPERPARAMETERS']['TAU'])
    epsilon_ini = float(config['DRL_HYPERPARAMETERS']['EPSILON_INI'])
    epsilon_fin = float(config['DRL_HYPERPARAMETERS']['EPSILON_FIN'])
    epsilon_step_percent = float(config['DRL_HYPERPARAMETERS']['EPSILON_STEP_PERCENT'])
    lr_actor = float(config['DRL_HYPERPARAMETERS']['LR_ACTOR'])
    lr_critic = float(config['DRL_HYPERPARAMETERS']['LR_CRITIC'])
    entropy = int(config['DRL_HYPERPARAMETERS']['ENTROPY'])
    entropy_factor = float(config['DRL_HYPERPARAMETERS']['ENTROPY_FACTOR'])


    params = {
        'weight_inference_time': weight_inference_time,
        'max_energy_credit': max_energy_credit,
        'max_inference_latency': max_inference_latency,
        'split_algorithm': split_algorithm,
        'rl_algorithm': rl_algorithm,
        'inference': inference,
        'n_episodes': n_episodes,
        'episode_duration': episode_duration,
        'time_interval': time_interval,
        'start_episode': start_episode,
        'compression_rates': compression_rates,
        'accuracy_decrease': accuracy_decrease,
        'weight_accuracy': weight_accuracy,
        'top1_flag': top1_flag,
        'top1_bins': top1_bins,
        'param_path': param_path,

        'n_hidden_layer': n_hidden_layer,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'target_update': target_update,
        'target_update_policy': target_update_policy,
        'lr': lr,
        'discount_factor': discount_factor,
        'tau': tau,
        'epsilon_ini': epsilon_ini,
        'epsilon_fin': epsilon_fin,
        'epsilon_step_percent': epsilon_step_percent,

        'lr_actor': lr_actor,
        'lr_critic': lr_critic,
        'entropy': entropy,
        'entropy_factor': entropy_factor

    }
    return params
