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
        # total 27 states for the production dataset
        # first 5 is for ue_bandwidth, ue_freq, ue_flops_cycle, energy_cost, power
        # then bandwidth, freqs, flops_cycle for each network node (excluding the ue)
        # then 6 is for flops per block
        # then energy_credit_consumed
        # the last 6 is for the radio channel conditions for UE mobility (speed, rsrp, rsrq, cqi, snr, state)
        # for ns-3 dataset (speed, rsrp, cqi, snr, tb_size, delay, tbler, ccqi, ndi, csinr, cthr, thr) i.e. 12
        self.n_states = 5 + (3 * (self.num_nodes-1)) + 6 + 1 + 12
        split_choices, split_indices = enumerate_action_space(self.allowed_splits, self.num_nodes,
                                                                        allow_empty_nodes=True)
        # extended action space to include the compression rates
        self.action_space, self.action_indices = extended_action_space(split_choices,
                                                                       self.scenario_params['compression_rates'])
        self.n_actions = len(self.action_space)
        self.agent_type = 'ddqn' if self.rl_algorithm == 1 else 'a2c'
        if self.agent_type == 'ddqn':
            self.agent = DDQNAgent(self.scenario_params, self.n_states, self.n_actions, self.allowed_splits,
                                   self.num_nodes, self.flops_per_block, split_indices)

        else:
            self.agent = A2CAgent(self.scenario_params, self.n_states, self.n_actions, self.allowed_splits,
                                   self.num_nodes, self.flops_per_block, split_indices)
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
        action = None
        action_idx = None
        self.episode_count = episode_count
        # define the agent attributes
        self.define_agent_attributes()
        # if training mode is on
        if not self.scenario_params['inference']:
            # train the agent based on the type of algorithm to run
            if self.agent_type == 'ddqn':
                action, action_idx = self.train_ddqn_agent(time, dnn_model, episode_params, output)
            else:
                action, action_idx = self.train_a2c_agent(time, dnn_model, episode_params, output)
        return action['split'], action['compression'], action_idx

    def train_a2c_agent(self, time, dnn_model, episode_params, output):
        # state is a vector, while log probs, rewards actions and entropies are scalars
        state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        state_cloned = state.clone()
        action, action_idx, _, _ = self.agent.choose_action(self.action_space, state_cloned)
        inference_time, ue_en_comp, ue_en_comm, top1_acc_conf = self.agent.perform_action(action, self.allowed_splits_blocks,
                                                                           dnn_model, episode_params, output)
        # extract index of full action that was SELECTED
        for k, v in self.action_indices.items():
            if v == action:
                action_idx = k
                break
        reward = self.agent.get_instant_reward(inference_time, ue_en_comp, ue_en_comm, top1_acc_conf)
        # log the reward
        self.agent.reward.append({'time': time, 'reward': reward})
        # update reward counter and compute cumulative average
        self.agent.reward_counter += 1
        self.agent.cumulative_reward = reward / self.agent.reward_counter
        next_state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        # print version
        #print('Initial version {}'.format(state._version))
        self.agent.replay_buffer.push(Sample(
            #log_prob.detach(),
            state,
            torch.tensor([reward]),
            next_state,
            torch.tensor([action_idx])
            #torch.tensor([entropy])
        ))
        # print version
        #print('Version after replay buffer {}'.format(state._version))
        if self.agent.replay_buffer.check_provide_samples(self.agent.batch_size):
            #print('Inside batch block')
            samples = self.agent.replay_buffer.sample(self.agent.batch_size)
            s, r, s_prime, act = extract_tensors(samples, 'sample')
            #print(r.size())
            probs = self.agent.actor(s)
            probs = probs + 1e-8
            dist = torch.distributions.Categorical(probs=probs)
            lp = dist.log_prob(act)
            entropy = dist.entropy()
            target = r.unsqueeze(1) + self.agent.discount_factor * self.agent.critic(s_prime)
            current = self.agent.critic(s)
            advantage = target - current
            #advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            lp = lp.unsqueeze(1)
            entropy = entropy.unsqueeze(1)

            critic_loss = advantage.pow(2).mean()
            actor_loss = torch.mean(-lp * advantage.detach() - self.agent.entropy * (
                    self.agent.entropy_factor * entropy.clone()))
            #actor_loss = torch.mean(-lp.clone() * advantage.detach())

            # ----- for debug ------
            #print('Initial version {}'.format(actor_loss._version))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            #print('Version before backward call {}'.format(actor_loss._version))
            #with torch.autograd.detect_anomaly():
            actor_loss.backward()
            critic_loss.backward()

            #print(actor_loss._version)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.agent.replay_buffer.clearSamples()
            # logging
            self.agent.actor_loss.append({'time': time, 'loss': actor_loss.item()})
            self.agent.critic_loss.append({'time': time, 'loss': critic_loss.item()})
            self.agent.advantages.append({'time': time, 'advantage': advantage.detach().mean().numpy()})
            self.agent.entropies.append({'time': time, 'entropy': entropy.detach().mean().numpy()})
        return action, action_idx

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
        # debug
        #print('state {}'.format(state))
        #print('action selected {}'.format(action))
        # this function also checks if the action selected is feasible or not
        inference_time, ue_en_comp, ue_en_comm, top1_acc_conf = self.agent.perform_action(action, self.allowed_splits_blocks,
                                                                           dnn_model, episode_params, output)
        # debug
        #print('action performed {}'.format(self.agent.split_compression_action))
        # extract index of full action that was SELECTED
        for k, v in self.action_indices.items():
            if v == action:
                action_idx = k
                break
        #print('Split config {}, success {}, n_success {}'.format(action, self.agent.success, self.agent.n_success))
        #print()
        reward = self.agent.get_instant_reward(inference_time, ue_en_comp, ue_en_comm, top1_acc_conf)
        # log the reward
        self.agent.reward.append({'time': time, 'reward': reward})
        # update reward counter and compute cumulative average
        self.agent.reward_counter += 1
        self.agent.cumulative_reward = reward / self.agent.reward_counter
        next_state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        #print('next state {}'.format(next_state))
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
            #print('predicted {}'.format(current_q_values))
            next_q_values = QValues.get_next_ddqn(self.agent, self.target_agent, s_prime)
            # normalize reward
            signed = torch.sign(r)
            r = signed * torch.log(1 + torch.abs(r))
            target_q_values = (next_q_values * self.agent.discount_factor) + r
            #print(current_q_values)
            #print('target {}'.format(target_q_values.unsqueeze(1)))
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

        return action, action_idx

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

