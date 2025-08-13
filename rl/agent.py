"""
agent.py

Defines the generic RL agent and its associated functions to train or infer the RL algorithm
"""

from rl.ddqn import DDQNAgent

class Agent:
    def __init__(self, scenario_params, allowed_splits, num_nodes):
        self.scenario_params = scenario_params
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes
        self.rl_algorithm = self.scenario_params['rl_algorithm']
        self.agent_type = 'ddqn' if self.rl_algorithm == 1 else 'a2c'
        if self.agent_type == 'ddqn':
            self.agent = DDQNAgent(self.scenario_params)


    def execute(self, episode_count, model, episode_params):
        # if training mode is on
        if self.scenario_params['inference'] == 0:
            epsilon = self.get_epsilon(episode_count)


    def get_epsilon(self, episode_count):
        return 1
