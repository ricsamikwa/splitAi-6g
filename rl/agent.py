"""
agent.py

Defines the generic RL agent and its associated methods to train or infer the RL algorithm
"""

from rl.ddqn import DDQNAgent, QValues
from rl.a2c import  A2CAgent
from rl.replay_buffer import Experience, extract_tensors, Sample
from utils.action_space import enumerate_action_space, extended_action_space
from utils.rl_utils import load_model_params

import torch
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)

class Agent:
    def __init__(self, scenario_params, allowed_splits, num_nodes, flops_per_block, allowed_splits_blocks):
        self.scenario_params = scenario_params
        self.allowed_splits = allowed_splits
        self.num_nodes = num_nodes
        self.flops_per_block = flops_per_block
        self.allowed_splits_blocks = allowed_splits_blocks
        self.rl_algorithm = self.scenario_params['rl_algorithm']
        # initializing class variables that are to be defined later
        self.target_agent = None
        self.episode_count = None
        self.optimizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        # total 27 states
        # first 5 is for ue_bandwidth, ue_freq, ue_flops_cycle, energy_cost, power
        # then bandwidth, freqs, flops_cycle for each network node (excluding the ue)
        # then 6 is for flops per block
        # then energy_credit_consumed
        # the last 6 is for the radio channel conditions for UE mobility (speed, rsrp, rsrq, cqi, snr, state)
        self.n_states = 5 + (3 * (self.num_nodes-1)) + 6 + 1 + 6
        self.action_space, self.action_indices = enumerate_action_space(self.allowed_splits, self.num_nodes,
                                                                        allow_empty_nodes=True)
        self.n_actions = len(self.action_space)
        self.agent_type = 'ddqn' if self.rl_algorithm == 1 else 'a2c'
        if self.agent_type == 'ddqn':
            self.agent = DDQNAgent(self.scenario_params, self.n_states, self.n_actions, self.allowed_splits,
                                   self.num_nodes, self.flops_per_block)

        else:
            self.agent = A2CAgent(self.scenario_params, self.n_states, self.n_actions, self.allowed_splits,
                                   self.num_nodes, self.flops_per_block)
            self.lr_actor = self.scenario_params['lr_actor']
            self.lr_critic = self.scenario_params['lr_critic']


    def execute(self, time, episode_count, dnn_model, episode_params, output):
        """
        Function that simulates the behavior of the agent.
        Args:
            time (int): The time step within the episode.
            episode_count (int): The episode number.
            dnn_model (pytorch model): The DNN model.
            episode_params (dict): The params specific to the episode packed in a dict.
            output (tensor): The pytorch tensor capturing the input image after transformation.

        Returns:
            The final split config to be used for inference.
        """
        self.episode_count = episode_count
        # define the agent attributes
        self.define_agent_attributes()
        # if training mode is on
        if not self.scenario_params['inference']:
            # train the agent based on the type of algorithm to run
            if self.agent_type == 'ddqn':
                self.train_ddqn_agent(time, dnn_model, episode_params, output)
            else:
                self.train_a2c_agent(time, dnn_model, episode_params, output)
        return self.agent.split_config

    def train_a2c_agent(self, time, dnn_model, episode_params, output):
        # state is a vector, while log probs, rewards actions and entropies are scalars
        state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        action, action_idx, entropy, log_prob = self.agent.choose_action(self.action_space, state)
        inference_time, ue_en_comp, ue_en_comm = self.agent.perform_action(action, self.allowed_splits_blocks,
                                                                           dnn_model, episode_params, output)
        reward = self.agent.get_instant_reward(inference_time, ue_en_comp, ue_en_comm)
        # log the reward
        self.agent.reward.append({'time': time, 'reward': reward})
        # update reward counter and compute cumulative average
        self.agent.reward_counter += 1
        self.agent.cumulative_reward = reward / self.agent.reward_counter
        next_state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        # print version
        self.agent.replay_buffer.push(Sample(
            log_prob,
            state,
            torch.tensor([reward]),
            next_state,
            torch.tensor([entropy])
        ))
        # print version
        if self.agent.replay_buffer.check_provide_samples(self.agent.batch_size):
            #print('Inside batch block')
            samples = self.agent.replay_buffer.sample(self.agent.batch_size)
            lp, s, r, s_prime, entropy = extract_tensors(samples, 'sample')

            target = r.unsqueeze(1) + self.agent.discount_factor * self.agent.critic(s_prime)
            current = self.agent.critic(s)
            advantage = target - current
            lp = lp.unsqueeze(1)
            entropy = entropy.unsqueeze(1)

            critic_loss = advantage.pow(2).mean()
            actor_loss = torch.mean(-lp * advantage.detach() - self.agent.entropy * (
                    self.agent.entropy_factor * entropy))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            #with torch.autograd.detect_anomaly():
            actor_loss.backward()
            critic_loss.backward()

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.agent.replay_buffer.clearSamples()
            # logging
            self.agent.actor_loss.append({'time': time, 'loss': actor_loss.item()})
            self.agent.critic_loss.append({'time': time, 'loss': critic_loss.item()})
            self.agent.advantages.append({'time': time, 'advantage': advantage.detach().mean().numpy()})


    def train_ddqn_agent(self, time, dnn_model, episode_params, output):
        """
        Function that trains the agent.
        Args:
            time (int): The time step within the episode.
            dnn_model (pytorch model): The DNN model.
            episode_params (dict): The params specific to the episode packed in a dict.
            output (tensor): The pytorch tensor capturing the input image after transformation.

        Returns:
            The final split config to be used for inference.
        """
        action_idx = None
        state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        action = self.agent.choose_action(self.action_space, state)
        # this function also checks and returns the feasible action
        inference_time, ue_en_comp, ue_en_comm = self.agent.perform_action(action, self.allowed_splits_blocks,
                                                                           dnn_model, episode_params, output)
        for k, v in self.action_indices.items():
            if v == self.agent.split_config:
                action_idx = k
                break
        # log the selected compression rate
        self.agent.selected_compression_rate.append({'time': time, 'compression': action['compression']})
        #print('Split config {}, success {}, n_success {}'.format(action, self.agent.success, self.agent.n_success))
        #print()
        reward = self.agent.get_instant_reward(inference_time, ue_en_comp, ue_en_comm)
        # log the reward
        self.agent.reward.append({'time': time, 'reward': reward})
        # update reward counter and compute cumulative average
        self.agent.reward_counter += 1
        self.agent.cumulative_reward = reward / self.agent.reward_counter
        next_state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        # collect the experience in the replay buffer
        self.agent.replay_buffer.push(Experience(
            state.clone().detach(), torch.tensor([action_idx]),next_state.clone().detach(), torch.tensor([reward])
        ))
        # if there are sufficient experiences in the replay buffer
        if self.agent.replay_buffer.check_provide_samples(self.agent.batch_size):
            experiences = self.agent.replay_buffer.sample(self.agent.batch_size)
            s, a, s_prime, r = extract_tensors(experiences, 'experience')
            # training mode
            current_q_values = QValues.get_current(self.agent, s, a)
            next_q_values = QValues.get_next_ddqn(self.agent, self.target_agent, s_prime)
            # normalize reward
            signed = torch.sign(r)
            r = signed * torch.log(1 + torch.abs(r))
            target_q_values = (next_q_values * self.agent.discount_factor) + r
            #print(current_q_values)
            #print(target_q_values.unsqueeze(1))
            criterion = torch.nn.SmoothL1Loss()
            #criterion = torch.nn.MSELoss()
            loss = criterion(current_q_values.float(), target_q_values.unsqueeze(1).float())
            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            for param in self.agent.parameters():
                param.grad.data.clamp_(-1, 1)
            # end of gradient clipping
            self.optimizer.step()
            # implement ONLY soft updates for now
            for target_param, local_param in zip(self.target_agent.parameters(),
                                                 self.agent.parameters()):
                target_param.data.copy_(
                    self.scenario_params['tau'] * local_param.data + (1 - self.scenario_params['tau']) * target_param.data)
            self.agent.loss_counter += 1
            if not self.agent.loss_counter % 50:
                print('Loss {}'.format(loss.item()))
            self.agent.loss.append({'time': time, 'loss': loss.item()})

    def define_agent_attributes(self):
        """
        Function that defines attributes specific to the agent algorithm.
        For ddqn, it defines the optimizer and initializes/loads the params of the target agent.
        Returns:

        """
        if self.agent_type == 'ddqn':
            # define optimizer for the ddqn agent
            self.optimizer = optim.Adam(params=self.agent.parameters(), lr=self.scenario_params['lr'])
            # load model
            self.agent.load_model(self.episode_count, 'main')
             # only for training mode
            if not self.scenario_params['inference']:
                if self.episode_count == 1:
                    # only for the first episode, the params of target nn are identical to policy (main) agent
                    self.target_agent = self.agent
                    self.target_agent.load_state_dict(self.agent.state_dict())
                else:
                    # for episodes > 1, load params of target agent from previous episode
                    self.target_agent = self.agent
                    self.target_agent.load_state_dict(load_model_params(self.agent_type, 'target',
                                                                        self.scenario_params,
                                                                        self.episode_count - 1))
                # set target agent to evaluation mode (no training)
                self.target_agent.eval()
        else:
            # both actor and critic need individual optimizers
            self.actor_optimizer = optim.Adam(params=self.agent.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(params=self.agent.critic.parameters(), lr=self.lr_critic)
            # then load models (inference case should be covered in this method)
            self.agent.load_model_a2c(self.episode_count)

