import torch
from utils.logging_utils import return_order, parse_episode_number

def load_model_params(agent_type, params, episode_count):
    order = return_order(params['n_episodes'])
    episode_count = parse_episode_number(order, episode_count)
    file = 'logs/{}/models/model_ep{}.pt'.format(agent_type, episode_count)
    return torch.load(file)

def save_model_params(model, agent_type, params, episode_count):
    if not params['variable_load']:
        file = 'logs/{}/models/model_ep{}.pt'.format(agent_type, episode_count)
    else:
        file = 'logs/{}/models/model_ep{}.pt'.format(agent_type, episode_count)
    torch.save(model.state_dict(), file)

