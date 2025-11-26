import csv
from csv import writer
from utils.rl_utils import save_model_params, return_order, parse_episode_number


def writeToCsv(data, filename, folder):
    with open('logs/{}/{}.csv'.format(folder, filename), 'w', encoding='utf8', newline='') as file:
        fc = csv.DictWriter(file, fieldnames=data[0].keys())
        fc.writeheader()
        fc.writerows(data)

def write_logs(scenario_params, episode, data, model):
    order = return_order(scenario_params['n_episodes'])
    episode_count = parse_episode_number(order, episode)
    if scenario_params['split_algorithm'] == 1:
        folder = 'random'
    elif scenario_params['split_algorithm'] == 2:
        if scenario_params['rl_algorithm'] == 1:
            folder = 'rl/ddqn'
        else:
            folder = 'rl/a2c'
    elif scenario_params['split_algorithm'] == 3:
        folder = 'optimum'
    elif scenario_params['split_algorithm'] == 4:
        folder = 'fixed'
    else:
        folder = 'ue'
    # inference time
    filename = 'system/{}_{}'.format('inference_time', episode_count)
    writeToCsv(data['inference_time'], filename, folder)
    # ue energy computation
    filename = 'system/{}_{}'.format('ue_energy_comp', episode_count)
    writeToCsv(data['ue_energy_comp'], filename, folder)
    # ue energy communication
    filename = 'system/{}_{}'.format('ue_energy_comm', episode_count)
    writeToCsv(data['ue_energy_comm'], filename, folder)
    # split config
    filename = 'splits/{}_{}'.format('split', episode_count)
    writeToCsv(data['split'], filename, folder)
    # top 1 accuracy
    filename = 'system/{}_{}'.format('top1', episode_count)
    writeToCsv(data['top1'], filename, folder)
    # top 5 accuracy
    #filename = 'system/{}_{}'.format('top5', episode_count)
    #writeToCsv(data['top5'], filename, folder)
    # energy credit
    filename = 'system/{}_{}'.format('energy_credit', episode_count)
    writeToCsv(data['energy_credit'], filename, folder)
    filename = 'system/{}_{}'.format('flops_off', episode_count)
    writeToCsv(data['flops_off'], filename, folder)
    if scenario_params['split_algorithm'] == 2:     # for rl algorithm
        # success rate
        #filename = 'system/{}_{}'.format('success_rate', episode_count)
        #writeToCsv(data['success_rate'], filename, folder)
        # total flops offloaded
        filename = 'system/{}_{}'.format('y_net', episode_count)
        writeToCsv(data['y_net'], filename, folder)
        # total flops on ue
        filename = 'system/{}_{}'.format('y_ue', episode_count)
        writeToCsv(data['y_ue'], filename, folder)

        if scenario_params['rl_algorithm'] == 1:    # ddqn
            save_model_params(model.agent, 'ddqn', 'main', scenario_params, episode_count)
            save_model_params(model.target_agent, 'ddqn', 'target', scenario_params, episode_count)
            filename = 'loss/loss_ep{}'.format(episode_count)
            writeToCsv(model.agent.loss, filename, folder)
            filename = 'reward/reward_ep{}'.format(episode_count)
            writeToCsv(model.agent.reward, filename, folder)
            fol = '{}/epsilon'.format(folder)
            logKPIs([model.agent.epsilon], 'epsilon', episode_count, fol)
        else:
            save_model_params(model.agent.actor, 'a2c', 'actor', scenario_params, episode_count)
            save_model_params(model.agent.critic, 'a2c', 'critic', scenario_params, episode_count)
            filename = 'loss/actor_loss_ep{}'.format(episode_count)
            writeToCsv(model.agent.actor_loss, filename, folder)
            filename = 'loss/critic_loss_ep{}'.format(episode_count)
            writeToCsv(model.agent.critic_loss, filename, folder)
            filename = 'advantage/advantage_ep{}'.format(episode_count)
            writeToCsv(model.agent.advantages, filename, folder)
            filename = 'reward/reward_ep{}'.format(episode_count)
            writeToCsv(model.agent.reward, filename, folder)


def logKPIs(data, kpi_type, episode_count, folder):
    file = 'logs/{}/{}.csv'.format(folder, kpi_type)
    with open(file, 'a', newline='') as f_object:
        if episode_count == 1:
            f_object.truncate(0)
        writer_object = writer(f_object)
        writer_object.writerow(data)
        f_object.close()

def read_single_col_data(filename, x, y, x_type, y_type):
    x_data = []
    y_data = []
    with open('{}.csv'.format(filename), 'r', encoding='utf8', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x_data.append(x_type(row[x]))
            y_data.append(y_type(row[y]))
    return x_data, y_data