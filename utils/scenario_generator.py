"""
scenario_generator.py

Utility function for generating all config parameters specific to the scenario

"""
import configparser

def generate_scenario():
    scenario_params = pack_parameters('config.ini')
    return scenario_params


def pack_parameters(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    # read all scenario parameters from config file
    split_algorithm = int(config['ALGORITHM']['SPLIT_ALGORITHM'])
    rl_algorithm = int(config['ALGORITHM']['RL_ALGORITHM'])
    inference = int(config['ALGORITHM']['INFERENCE'])
    n_episodes = int(config['ALGORITHM']['N_EPISODES'])
    episode_duration = int(config['ALGORITHM']['EPISODE_DURATION'])
    time_interval = int(config['ALGORITHM']['TIME_INTERVAL'])
    start_episode = int(config['ALGORITHM']['START_EPISODE'])

    n_hidden_layer = int(config['DRL_HYPERPARAMETERS']['N_HIDDEN_LAYER'])
    batch_size = int(config['DRL_HYPERPARAMETERS']['BATCH_SIZE'])
    buffer_size = int(config['DRL_HYPERPARAMETERS']['BUFFER_SIZE'])
    target_update = int(config['DRL_HYPERPARAMETERS']['TARGET_UPDATE'])
    target_update_policy = int(config['DRL_HYPERPARAMETERS']['TARGET_UPDATE_POLICY'])
    lr = float(config['DRL_HYPERPARAMETERS']['LR'])
    gamma = float(config['DRL_HYPERPARAMETERS']['GAMMA'])
    tau = float(config['DRL_HYPERPARAMETERS']['TAU'])
    epsilon_ini = float(config['DRL_HYPERPARAMETERS']['EPSILON_INI'])
    epsilon_fin = float(config['DRL_HYPERPARAMETERS']['EPSILON_FIN'])
    epsilon_step_percent = float(config['DRL_HYPERPARAMETERS']['EPSILON_STEP_PERCENT'])


    params = {
        'split_algorithm': split_algorithm,
        'rl_algorithm': rl_algorithm,
        'inference': inference,
        'n_episodes': n_episodes,
        'episode_duration': episode_duration,
        'time_interval': time_interval,
        'start_episode': start_episode,

        'n_hidden_layer': n_hidden_layer,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'target_update': target_update,
        'target_update_policy': target_update_policy,
        'lr': lr,
        'gamma': gamma,
        'tau': tau,
        'epsilon_ini': epsilon_ini,
        'epsilon_fin': epsilon_fin,
        'epsilon_step_percent': epsilon_step_percent,

    }
    return params
