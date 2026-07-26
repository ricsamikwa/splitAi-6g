from collections import namedtuple
import random
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
            #self.memory[self.push_count] = experience
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count = self.push_count + 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def check_provide_samples(self, batch_size):
        return len(self.memory) >= batch_size

    def clearSamples(self):
        self.memory.clear()

    def get_all(self):
        """
        Returns the buffer's contents in insertion (chronological) order. DDQN and A2C only ever need a
        random subsample via sample(), but PPO's GAE computation needs the full rollout in the order it
        was collected, since it bootstraps advantages backwards through consecutive transitions.

        Ordering is only guaranteed while len(memory) has not reached `capacity` since the last
        clearSamples() call: push() falls back to overwriting by (push_count % capacity) once the buffer
        is full, which can break chronological order. In practice this means the buffer's capacity
        (scenario_params['buffer_size']) should be sized to be >= the rollout length used between two
        PPO update()/clearSamples() calls.
        """
        return list(self.memory)


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

Sample = namedtuple('Sample', ('state_value', 'reward', 'next_state_value', 'action'))

# PPO needs the raw state (re-forward-passed through the actor across multiple epochs, unlike A2C which
# trains on a single pass) and the value estimate cached at action-selection time, plus the
# episode-termination flag needed to mask bootstrapping in GAE. Unlike an earlier version of this buffer,
# log_prob is NOT stored here: carrying an autograd-attached tensor from choose_action() across the buffer
# boundary into a later update() risks stale-graph issues (the same class of problem encountered with A2C).
# Instead, the "old" log-probs PPO's ratio needs are recomputed fresh at the start of update(), directly
# from (state, action), using the actor's pre-update weights - which are identical to the policy that
# generated the rollout, since no gradient step has occurred yet.
# action/value/reward/done are stored as 1-element tensors (not 0-dim scalars) so extract_tensors can
# torch.cat them, matching the existing convention used for Experience.action and Sample.action/reward.
PPOExperience = namedtuple('PPOExperience', ('state', 'action', 'value', 'reward', 'done'))


def extract_tensors(data, type):
    if type == 'experience':
        batch = Experience(*zip(*data))
        t1 = torch.stack(batch.state, dim=0)
        t2 = torch.cat(batch.action)
        t3 = torch.stack(batch.next_state, dim=0)
        t4 = torch.cat(batch.reward)
        return t1, t2, t3, t4
    elif type == 'ppo':
        batch = PPOExperience(*zip(*data))
        t1 = torch.stack(batch.state, dim=0)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.value)
        t4 = torch.cat(batch.reward)
        t5 = torch.cat(batch.done)
        return t1, t2, t3, t4, t5
    else:
        batch = Sample(*zip(*data))
        #t1 = torch.stack(batch.log_prob, dim=0)
        t2 = torch.stack(batch.state_value, dim=0)
        t3 = torch.cat(batch.reward)
        t4 = torch.stack(batch.next_state_value, dim=0)
        #t5 = torch.cat(batch.entropy)
        t6 = torch.cat(batch.action)
        return t2, t3, t4, t6



