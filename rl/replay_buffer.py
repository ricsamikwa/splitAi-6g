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


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

Sample = namedtuple('Sample', ('log_prob', 'state_value', 'reward', 'next_state_value', 'entropy'))



def extract_tensors(data, type):
    if type == 'experience':
        batch = Experience(*zip(*data))
        t1 = torch.stack(batch.state, dim=0)
        t2 = torch.cat(batch.action)
        t3 = torch.stack(batch.next_state, dim=0)
        t4 = torch.cat(batch.reward)
        return t1, t2, t3, t4
    else:
        batch = Sample(*zip(*data))
        t1 = torch.stack(batch.log_prob, dim=0)
        t2 = torch.stack(batch.state_value, dim=0)
        t3 = torch.cat(batch.reward)
        t4 = torch.stack(batch.next_state_value, dim=0)
        t5 = torch.cat(batch.entropy)
        return t1, t2, t3, t4, t5



