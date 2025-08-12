"""
agent.py

Defines the generic RL agent and its associated functions to train or infer the RL algorithm
"""

from rl.ddqn import DDQNAgent
class Agent:
    def __init__(self, scenario_params, allowed_splits, num_nodes):
        self.scenario_params = scenario_params
        self.rl_algorithm = self.scenario_params['rl_algorithm']
        self.agent_type = 'ddqn' if self.rl_algorithm == 1 else 'a2c'
        if self.agent_type == 'ddqn':
            self.agent = DDQNAgent(self.scenario_params)
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes

    def execute(self, episode_count):
        # if training mode is on
        if self.scenario_params['inference'] == 0:
            state = self.agent.get_agent_state()
            action = self.agent.choose_action()
            reward = self.agent.get_instant_reward()
            next_state = self.agent.get_agent_state()
