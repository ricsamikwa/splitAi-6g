"""
agent.py

Defines the generic RL agent and its associated methods to train or infer the RL algorithm
"""
from rl.ddqn import DDQNAgent, QValues
from rl.a2c import A2CAgent
from rl.ppo import PPOAgent
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
        # rl_algorithm: 1 -> ddqn, 2 -> a2c, 3 -> ppo. Kept as explicit branches (rather than an
        # 'else' catch-all as before) now that there are three algorithms to distinguish.
        if self.rl_algorithm == 1:
            self.agent_type = 'ddqn'
        elif self.rl_algorithm == 2:
            self.agent_type = 'a2c'
        else:
            self.agent_type = 'ppo'

        if self.agent_type == 'ddqn':
            self.agent = DDQNAgent(self.scenario_params, self.n_states, self.n_actions, self.allowed_splits,
                                   self.num_nodes, self.flops_per_block, split_indices)

        elif self.agent_type == 'a2c':
            self.agent = A2CAgent(self.scenario_params, self.n_states, self.n_actions, self.allowed_splits,
                                   self.num_nodes, self.flops_per_block, split_indices)
            self.lr_actor = self.scenario_params['lr_actor']
            self.lr_critic = self.scenario_params['lr_critic']

        else:
            # PPOAgent's Actor/Critic each build and own their own optimizer at construction time (from
            # scenario_params['ppo_lr_actor'] / ['ppo_lr_critic']), so - unlike a2c - no separate lr_actor/
            # lr_critic bookkeeping is needed here; define_agent_attributes() does not need to (re)create
            # optimizers for ppo either, see below.
            self.agent = PPOAgent(self.scenario_params, self.n_states, self.n_actions, self.allowed_splits,
                                   self.num_nodes, self.flops_per_block, split_indices)


    def execute(self, time, episode_count, dnn_model, episode_params, output, done=False):
        """
        Function that simulates the behavior of the agent.
        Args:
            time (int): The time step within the episode.
            episode_count (int): The episode number.
            dnn_model (pytorch model): The DNN model.
            episode_params (dict): The params specific to the episode packed in a dict.
            output (tensor): The pytorch tensor capturing the input image after transformation.
            done (bool): Whether this call corresponds to the final time step of the episode. Only
                consumed by PPO (its on-policy rollout needs to know episode boundaries to correctly mask
                bootstrapping in GAE - see ppo.py). Ignored by ddqn/a2c, which do not need it. The caller
                (the outer per-episode time loop) already knows when it is on the last time step, so it is
                the natural place to supply this rather than inferring it here from scenario_params.

        Returns:
            The final split config and the top1 acc confidence to be used for inference.
        """
        action = None
        action_idx = None
        top1_acc_conf = None
        self.episode_count = episode_count
        # define the agent attributes
        self.define_agent_attributes()
        # if training mode is on
        if not self.scenario_params['inference']:
            # train the agent based on the type of algorithm to run
            if self.agent_type == 'ddqn':
                action, action_idx, top1_acc_conf = self.train_ddqn_agent(time, dnn_model, episode_params, output)
            elif self.agent_type == 'a2c':
                action, action_idx, top1_acc_conf = self.train_a2c_agent(time, dnn_model, episode_params, output)
            else:
                action, action_idx, top1_acc_conf = self.train_ppo_agent(time, dnn_model, episode_params, output, done)
        else:
            # implement inference here, no training, just return the selected action and top1 due to the checkpoint policy
            state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        return action['split'], action['compression'], action_idx, top1_acc_conf

    def train_a2c_agent(self, time, dnn_model, episode_params, output):
        # state is a vector, while log probs, rewards actions and entropies are scalars
        state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        state_cloned = state.clone()
        action, action_idx = self.agent.choose_action(self.action_space, state_cloned)
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
        return action, action_idx, top1_acc_conf

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
            #signed = torch.sign(r)
            #r = signed * torch.log(1 + torch.abs(r))
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

        return action, action_idx, top1_acc_conf

    def train_ppo_agent(self, time, dnn_model, episode_params, output, done=False):
        """
        Function that trains the agent using the PPO algorithm.

        Unlike train_ddqn_agent/train_a2c_agent, this method does not manually manage the replay buffer,
        loss computation, or optimizer steps inline - PPOAgent (rl/ppo.py) encapsulates all of that behind
        store_transition() and update(), since PPO's update needs a full chronologically-ordered rollout
        (for GAE) and multiple epochs over it, rather than a single fixed-size random minibatch like a2c's
        inline training above. This method's job is just to: act, log the transition, and trigger update()
        once enough on-policy samples have been collected.

        Args:
            time (int): The time step within the episode.
            dnn_model (pytorch model): The DNN model.
            episode_params (dict): The params specific to the episode packed in a dict.
            output (tensor): The pytorch tensor capturing the input image after transformation.
            done (bool): Whether this is the final time step of the current episode (see execute()).

        Returns:
            The final split config, its index, and the top1 acc confidence to be used for inference.
            As a side effect, when an update is triggered, appends {'time': time, 'loss': ...} entries to
            self.agent.actor_loss / self.agent.critic_loss, matching train_a2c_agent's logging convention.
        """
        state = self.agent.get_agent_state(episode_params, self.flops_per_block)
        action, action_idx = self.agent.choose_action(self.action_space, state)
        inference_time, ue_en_comp, ue_en_comm, top1_acc_conf = self.agent.perform_action(action, self.allowed_splits_blocks,
                                                                           dnn_model, episode_params, output)
        # extract index of full action that was SELECTED (action_idx from choose_action indexes into the
        # feasible action_space passed in, not the global action_indices mapping - same reasoning as a2c
        # and ddqn above)
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

        # push (state, action, value, reward, done) into the on-policy rollout buffer; log_prob is
        # deliberately not passed here (or stored at all) - it is recomputed fresh inside update() from the
        # actor's pre-update weights, to avoid carrying an autograd-attached tensor across this call
        # boundary (see rl/ppo.py's module docstring)
        self.agent.store_transition(reward, done)

        # trigger a PPO update once the on-policy rollout has reached the configured length. Falls back to
        # the buffer's full capacity if 'ppo_rollout_length' isn't set in scenario_params, but a dedicated,
        # smaller rollout length (e.g. 128-2048, decoupled from ddqn's much larger off-policy buffer_size)
        # is recommended - see rl/ppo.py and ReplayBuffer.get_all()'s docstring for why the buffer's
        # capacity must be sized to match whatever rollout length is used here.
        rollout_length = self.scenario_params.get('ppo_rollout_length', self.agent.replay_buffer.capacity)
        if self.agent.replay_buffer.check_provide_samples(rollout_length):
            if done:
                # episode boundary: nothing to bootstrap from
                last_value = torch.zeros(1)
            else:
                next_state = self.agent.get_agent_state(episode_params, self.flops_per_block)
                with torch.no_grad():
                    last_value = self.agent.critic(next_state.clone().detach().float())
            actor_loss_val, critic_loss_val = self.agent.update(last_value)
            # log actor/critic loss for this time step, mirroring a2c's logging convention above. update()
            # returns (None, None) if the buffer somehow ended up empty (shouldn't happen given the
            # check_provide_samples guard, but guarded against here rather than logging a None loss)
            if actor_loss_val is not None:
                self.agent.actor_loss.append({'time': time, 'loss': actor_loss_val})
                self.agent.critic_loss.append({'time': time, 'loss': critic_loss_val})

        return action, action_idx, top1_acc_conf

    def define_agent_attributes(self):
        """
        Function that defines attributes specific to the agent algorithm.
        For ddqn, it defines the optimizer and initializes/loads the params of the target agent.
        For a2c, it defines the actor/critic optimizers and loads the model.
        For ppo, it only loads the model - PPOAgent's Actor/Critic already own their optimizers (built at
        construction time from scenario_params['ppo_lr_actor']/['ppo_lr_critic']), so there is nothing to
        (re)create here every call, unlike a2c.
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
        elif self.agent_type == 'a2c':
            # both actor and critic need individual optimizers
            self.actor_optimizer = optim.Adam(params=self.agent.actor.parameters(), lr=self.lr_actor)
            self.critic_optimizer = optim.Adam(params=self.agent.critic.parameters(), lr=self.lr_critic)
            # then load models (inference case should be covered in this method)
            self.agent.load_model_a2c(self.episode_count)
        else:
            # ppo: optimizers already live on self.agent.actor/self.agent.critic (see class docstring
            # above); only the model params need loading/restoring here, mirroring load_model_a2c.
            self.agent.load_model_ppo(self.episode_count)