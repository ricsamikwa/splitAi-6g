"""
ppo.py

Defines the RL agent running the PPO algorithm and its associated parameters to train or infer the RL algorithm.

Inherits from DDQNAgent (as A2CAgent does) so that state encoding, split-compression execution, reward
computation, energy-credit/latency/accuracy checks, and logging counters are reused as-is. Only the policy
representation (Actor/Critic), action selection, and the on-policy update rule are PPO-specific.

Notes on integration (mirrors A2CAgent's contract):
    - choose_action(playable_actions, state) returns (selected_split_compression, action_idx), exactly like
      A2CAgent, so existing call sites do not need branching logic beyond agent type.
    - Only the value estimate needed for GAE is cached at selection time (self._last_value) and picked up
      by store_transition(reward, done); log-probs are deliberately NOT cached/passed between calls - PPO's
      "old" log-probs are recomputed fresh at the start of update(), directly from the stored (state,
      action) pairs, using the actor's pre-update weights. This avoids carrying an autograd-attached tensor
      across the buffer boundary (the class of stale-graph issue previously seen with A2C).
    - Transitions are pushed as PPOExperience tuples into the inherited self.replay_buffer (same object
      DDQNAgent/A2CAgent use as their "Data Buffer"). Unlike DDQN's off-policy replay, PPO reads the buffer
      back with get_all() (chronological order, required for GAE) rather than sample() (random subsample),
      and clears it after every update() since on-policy samples cannot be reused once the policy moves.
      IMPORTANT: for get_all()'s ordering guarantee to hold, scenario_params['buffer_size'] must be >= the
      rollout length collected between two update() calls - see ReplayBuffer.get_all()'s docstring.
    - update() runs the clipped-surrogate PPO update over the collected rollout using Generalized Advantage
      Estimation (GAE), for scenario_params['ppo_epochs'] epochs over minibatches of size
      scenario_params['ppo_minibatch_size'].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rl_utils import load_model_params
from rl.ddqn import DDQNAgent
from rl.replay_buffer import PPOExperience, extract_tensors

PPO_ACTOR_CHECKPOINT = 'ppo_actor_inference.pt'
PPO_CRITIC_CHECKPOINT = 'ppo_critic_inference.pt'

class PPOAgent(DDQNAgent, nn.Module):
    def __init__(self, scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block, split_indices):
        DDQNAgent.__init__(self, scenario_params, n_states, n_actions, allowed_splits, num_nodes, flops_per_block, split_indices)
        nn.Module.__init__(self)
        self.actor = Actor(self.n_states, self.n_actions, self.scenario_params)
        self.critic = Critic(self.n_states, self.scenario_params)

        # PPO-specific hyperparameters (fall back to sensible defaults if not present in scenario_params,
        # so existing config files used for ddqn/a2c do not need to be touched to smoke-test this agent)
        self.clip_epsilon = self.scenario_params.get('ppo_clip_epsilon', 0.2)
        self.gae_lambda = self.scenario_params.get('ppo_gae_lambda', 0.95)
        self.ppo_epochs = self.scenario_params.get('ppo_epochs', 4)
        self.ppo_minibatch_size = self.scenario_params.get('ppo_minibatch_size', self.batch_size)
        self.value_loss_coef = self.scenario_params.get('ppo_value_loss_coef', 0.5)
        self.entropy_factor = self.scenario_params.get('entropy_factor', 0.005)
        self.max_grad_norm = self.scenario_params.get('ppo_max_grad_norm', 0.5)

        # NOTE: DDQNAgent.__init__ already instantiated self.replay_buffer (rl/replay_buffer.py,
        # capacity=scenario_params['buffer_size']). PPO reuses it as its on-policy rollout buffer, storing
        # PPOExperience tuples instead of DDQN's Experience/A2C's Sample tuples. See module docstring for
        # the capacity caveat.

        # cached from the most recent choose_action() call, consumed by store_transition()
        self._last_state = None
        self._last_action_idx = None
        self._last_value = None

        # logging, kept consistent with the naming used by DDQNAgent/A2CAgent. actor_loss/critic_loss are
        # populated externally (by agent.py, after each update() call) as {'time': ..., 'loss': ...} dicts,
        # matching a2c's convention - update() itself only returns the mean loss values, it does not
        # self-append to these lists. entropies is still appended to internally, per minibatch, in update().
        self.actor_loss = []
        self.critic_loss = []
        self.entropies = []

    def load_model_ppo(self, episode_count):
        if not self.scenario_params['inference']:
            if episode_count > 1:
                agent = load_model_params('ppo', 'actor', self.scenario_params, episode_count - 1)
                self.actor.load_state_dict(agent)
                agent = load_model_params('ppo', 'critic', self.scenario_params, episode_count - 1)
                self.critic.load_state_dict(agent)
            else:
                self.actor.load_state_dict(torch.load('rl/initial_models/actor_params_ppo.pt'))
                self.critic.load_state_dict(torch.load('rl/initial_models/critic_params_ppo.pt'))
        else:
            self.actor.load_state_dict(torch.load('rl/inference_checkpoints/{}'.format(PPO_ACTOR_CHECKPOINT)))
            self.critic.load_state_dict(torch.load('rl/inference_checkpoints/{}'.format(PPO_CRITIC_CHECKPOINT)))

    def choose_action(self, playable_actions, state):
        state_t = state.clone().detach().float()
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs=probs)

        if not self.scenario_params['inference']:
            if len(playable_actions) == 1:
                action_idx = torch.tensor(0)
            else:
                action_idx = dist.sample()
        else:
            # greedy at inference time, same convention as DDQNAgent/A2CAgent
            if len(playable_actions) == 1:
                action_idx = torch.tensor(0)
            else:
                action_idx = probs.argmax()

        # cache value estimate for this state; picked up by store_transition() once the environment step's
        # reward is available (training only). log_prob is intentionally NOT cached here - see module
        # docstring; it's recomputed fresh in update() instead.
        if not self.scenario_params['inference']:
            with torch.no_grad():
                self._last_value = self.critic(state_t)  # shape (1,), left unsqueezed for buffer storage
            self._last_state = state_t
            self._last_action_idx = action_idx

        selected_split_compression = playable_actions[action_idx]
        return selected_split_compression, action_idx

    def store_transition(self, reward, done):
        """
        Pushes the cached (state, action, value) from the last choose_action() call, together with the
        observed reward and episode-termination flag, into self.replay_buffer as a PPOExperience. Mirrors
        how DDQN/A2C push a completed (s, a, r, s') tuple once the environment step's reward is known.
        action/reward/done are reshaped to 1-element tensors to match the buffer's torch.cat convention
        (see rl/replay_buffer.py).
        """
        experience = PPOExperience(
            self._last_state,
            self._last_action_idx.view(1),
            self._last_value.view(1),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([float(done)], dtype=torch.float32),
        )
        self.replay_buffer.push(experience)

    def compute_gae(self, values, rewards, dones, last_value):
        """
        Generalized Advantage Estimation (Schulman et al., "High-Dimensional Continuous Control Using
        Generalized Advantage Estimation"), addressing the bias-variance tradeoff of the plain n-step
        advantage used in ENSPLIT-2 (A2C).

        values, rewards, dones: 1-D tensors aligned with the rollout, in chronological order.
        last_value: bootstrap value (1-element tensor) of the state following the final stored transition
            (zero if terminal).
        Returns: (advantages, returns) as 1-D tensors aligned with the rollout.
        """
        values_ext = torch.cat([values, last_value])
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.discount_factor * values_ext[t + 1] * mask - values_ext[t]
            gae = delta + self.discount_factor * self.gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, last_value=None):
        """
        Runs the clipped-surrogate PPO update over the entire collected rollout (read back in chronological
        order via self.replay_buffer.get_all()) for self.ppo_epochs epochs, in minibatches of
        self.ppo_minibatch_size, then clears the buffer (on-policy: samples cannot be reused once the
        policy that generated them has changed).

        Callers should invoke this once enough transitions have accumulated (analogous to the buffer-size
        trigger used to train A2C, e.g. via self.replay_buffer.check_provide_samples(...)), passing the
        bootstrap value of the state after the last stored transition (or leave as None to treat it as
        terminal, i.e. bootstrap value 0).

        Returns:
            (mean_actor_loss, mean_critic_loss): floats averaged over every epoch/minibatch step in this
            update call, or (None, None) if there was nothing in the buffer to train on. Deliberately
            returned rather than self-appended here (unlike self.entropies, which still logs internally per
            minibatch) - the caller (agent.py) knows the current 'time' step and tags/appends these into
            self.actor_loss/self.critic_loss itself, mirroring a2c's {'time': ..., 'loss': ...} logging
            convention in agent.py rather than duplicating a second, differently-shaped logging path here.
        """
        data = self.replay_buffer.get_all()
        if len(data) == 0:
            return None, None

        if last_value is None:
            last_value = torch.zeros(1)

        states, actions, values, rewards, dones = extract_tensors(data, 'ppo')

        # "Old" log-probs (the policy that generated this rollout) are recomputed here rather than carried
        # from choose_action(): at this point in update(), no gradient step has been taken yet, so
        # self.actor's weights are identical to the rollout-collection policy - recomputing gives the same
        # values as storing would have, without keeping an autograd-attached tensor alive across the buffer
        # boundary (the class of stale-graph issue previously seen with A2C).
        with torch.no_grad():
            old_probs = self.actor(states)
            old_dist = torch.distributions.Categorical(probs=old_probs)
            old_log_probs = old_dist.log_prob(actions)

        advantages, returns = self.compute_gae(values, rewards, dones, last_value)
        # normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = states.shape[0]
        indices = np.arange(n_samples)
        actor_loss_history = []
        critic_loss_history = []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, self.ppo_minibatch_size):
                end = start + self.ppo_minibatch_size
                mb_idx = indices[start:end]
                if len(mb_idx) == 0:
                    continue
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long)

                mb_states = states[mb_idx_t]
                mb_actions = actions[mb_idx_t]
                mb_old_log_probs = old_log_probs[mb_idx_t]
                mb_advantages = advantages[mb_idx_t]
                mb_returns = returns[mb_idx_t]

                probs = self.actor(mb_states)
                dist = torch.distributions.Categorical(probs=probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_factor * entropy

                values_pred = self.critic(mb_states).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, mb_returns)

                loss = actor_loss + self.value_loss_coef * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                actor_loss_history.append(actor_loss.item())
                critic_loss_history.append(critic_loss.item())
                self.entropies.append(entropy.item())

        self.replay_buffer.clearSamples()

        mean_actor_loss = float(np.mean(actor_loss_history)) if actor_loss_history else None
        mean_critic_loss = float(np.mean(critic_loss_history)) if critic_loss_history else None
        return mean_actor_loss, mean_critic_loss


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, scenario_params):
        nn.Module.__init__(self)
        self.scenario_params = scenario_params
        self.model = nn.Sequential(nn.Linear(n_states, 256), nn.Tanh(),
                                    nn.Linear(256, 256), nn.Tanh(),
                                    nn.Linear(256, n_actions),
                                    )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=scenario_params.get('ppo_lr_actor', 1e-4))

    def forward(self, x):
        y = self.model(x)
        y = F.softmax(y, dim=-1)
        return y


class Critic(nn.Module):
    def __init__(self, n_states, scenario_params):
        nn.Module.__init__(self)
        self.scenario_params = scenario_params
        self.layer1 = nn.Linear(n_states, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=scenario_params.get('ppo_lr_critic', 5e-4))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)