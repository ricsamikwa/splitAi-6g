import torch
from utils.logging_utils import return_order, parse_episode_number

def load_model_params(agent_type, nn_type, scenario_params, episode_count):
    """
    Script to load pytorch model parameters from file.
    Args:
        agent_type (str): indicates the RL algorithm e.g. DDQN / A2C / PPO
        nn_type (str): indicates the type of nn i.e. policy nn or target nn
        scenario_params (dict): dict storing the config parameters related to the scenario
        episode_count (int): the episode number

    Returns:
        the parameters of the indicated pytorch model that have been read from disk
    """
    order = return_order(scenario_params['n_episodes'])
    episode_count = parse_episode_number(order, episode_count)
    file = 'logs/rl/{}/models/model_{}_ep{}.pt'.format(agent_type, nn_type, episode_count)
    return torch.load(file)

def save_model_params(model, agent_type, nn_type, scenario_params, episode_count):
    """
    Script to save model parameters on disk.
    Args:
        model (pytorch object): indicates the state of the pytorch object
        agent_type (str): indicates the RL algorithm e.g. DDQN / A2C / PPO
        nn_type (str): indicates the type of nn i.e. policy nn or target nn
        scenario_params (dict): dict storing the config parameters related to the scenario
        episode_count (int): the episode number

    Returns:

    """
    file = 'logs/rl/{}/models/model_{}_ep{}.pt'.format(agent_type, nn_type, episode_count)
    torch.save(model.state_dict(), file)

