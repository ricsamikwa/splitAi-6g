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
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def check_provide_samples(self, batch_size):
        return len(self.memory) >= batch_size

    def clearSamples(self):
        self.memory.clear()


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


def extract_tensors(data):
    batch = Experience(*zip(*data))
    t1 = torch.stack(batch.state, dim=0)
    t2 = torch.cat(batch.action)
    t3 = torch.stack(batch.next_state, dim=0)
    t4 = torch.cat(batch.reward)
    return t1, t2, t3, t4



