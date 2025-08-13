import csv

def return_order(n_episodes):
    f = 1
    order = None
    while f >= 1:
        if 10 ** f > n_episodes:
            order = f
            f = 0
        else:
            f = f + 1
            continue
    return order


def parse_episode_number(order, ep):
    possible_orders = [o for o in range(order, -1, -1)]
    n_zeros = None
    for o in possible_orders:
        if 10 ** o > ep >= 10 ** (o - 1):
            n_zeros = order - o
            break
    if n_zeros == 0:
        return str(ep)
    else:
        s = ''
        for i in range(n_zeros):
            s = s + '0'
        return s + str(ep)

def writeToCsv(data, filename, folder):
    with open('logs/{}/{}.csv'.format(folder, filename), 'w', encoding='utf8', newline='') as file:
        fc = csv.DictWriter(file, fieldnames=data[0].keys())
        fc.writeheader()
        fc.writerows(data)

def write_logs(scenario_params, episode, data_name, data):
    order = return_order(scenario_params['n_episodes'])
    episode_count = parse_episode_number(order, episode)
    if scenario_params['split_algorithm'] == 1:
        folder = 'random'
    else:
        folder = 'rl'
    filename = '{}_{}'.format(data_name, episode_count)
    writeToCsv(data, filename, folder)