import torch

from collections import deque

from src.human import retrieve_human_buffer

import random

import itertools

import numpy as np
from collections import namedtuple

class ReplayMemory:
    def __init__(self, device, size=1000, batch_size=32):
        self._buffer: deque = deque(maxlen=size)
        self._batch_size = batch_size
        self._device = device

    def push(self, transition):
        s, a, r, s_, d = transition

        # move to cpu
        self._buffer.append((s.cpu(), a, r, s_.cpu(), d))

    def sample(self):
        # sample
        s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))

        # reshape, convert if needed, put on device
        return (
            torch.cat(s, 0).to(self._device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self._device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self._device),
            torch.cat(s_, 0).to(self._device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(self._device),
        )

    def __len__(self) -> int:
        return len(self._buffer)

## ---------------------------------------------------------


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class PrioritizedReplayMemory:
    def __init__(self, device, size=1000, batch_size=32, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        self._device = device
        self._size = size
        self._batch_size = batch_size
        self._alpha = alpha
        self._beta = beta
        self._beta_increment = beta_increment
        self._epsilon = epsilon
        
        self._buffer = []
        self._priorities = np.zeros(size, dtype=np.float32)
        self._position = 0
        self._total_transitions = 0

    def push(self, transition):
        s, a, r, s_, d = transition
        transition = (s.cpu(), a, r, s_.cpu(), d)
        
        if len(self._buffer) < self._size:
            self._buffer.append(None)
        
        self._buffer[self._position] = Transition(*transition)
        
        max_priority = self._priorities.max() if self._total_transitions > 0 else 1.0
        self._priorities[self._position] = max_priority
        
        self._position = (self._position + 1) % self._size
        self._total_transitions = min(self._total_transitions + 1, self._size)

    def sample(self):
        if self._total_transitions == 0:
            raise RuntimeError("Cannot sample from empty buffer")
        
        probs = self._priorities[:self._total_transitions] ** self._alpha
        probs /= probs.sum()
        
        indices = np.random.choice(
            self._total_transitions, 
            self._batch_size, 
            p=probs,
            replace=True
        )
        
        samples = [self._buffer[idx] for idx in indices]
        
        weights = (self._total_transitions * probs[indices]) ** (-self._beta)
        weights /= weights.max()  # Normalize weights
        
        self._beta = min(1.0, self._beta + self._beta_increment)
        
        states = torch.cat([s.state for s in samples], 0).to(self._device)
        actions = torch.tensor([s.action for s in samples], dtype=torch.int64).unsqueeze(1).to(self._device)
        rewards = torch.tensor([s.reward for s in samples], dtype=torch.float32).unsqueeze(1).to(self._device)
        next_states = torch.cat([s.next_state for s in samples], 0).to(self._device)
        dones = torch.tensor([s.done for s in samples], dtype=torch.uint8).unsqueeze(1).to(self._device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self._device)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self._priorities[idx] = abs(error) + self._epsilon

    def __len__(self):
        return self._total_transitions

## ---------------------------------------------------------

class HumanReplayMemory:
    def __init__(self, opt, size=1000, batch_size=32, human_records="_human"):
        self._buffer: deque = deque(maxlen=size)
        self._batch_size = batch_size
        self._device = opt.device

        self._human_buffer = retrieve_human_buffer(human_records, opt)

        self._epsilon = get_epsilon_schedule(
            start=0.9, end=0.1, steps=opt.steps * 0.5
        )

    def push(self, transition):
        s, a, r, s_, d = transition
        self._buffer.append((s.cpu(), a, r, s_.cpu(), d))

    def sample(self):
        if next(self._epsilon) < torch.rand(1).item():
            s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))
        else:
            s, a, r, s_, d = zip(*random.sample(self._human_buffer, self._batch_size))

        return (
            torch.cat(s, 0).to(self._device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self._device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self._device),
            torch.cat(s_, 0).to(self._device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(self._device),
        )

    def __len__(self) -> int:
        return len(self._buffer)

def get_epsilon_schedule(start=1.0, end=0.1, steps=500):
    """Returns either:
    - a generator of epsilon values
    - a function that receives the current step and returns an epsilon

    The epsilon values returned by the generator or function need
    to be degraded from the `start` value to the `end` within the number
    of `steps` and then continue returning the `end` value indefinetly.

    You can pick any schedule (exp, poly, etc.). I tested with linear decay.
    """
    eps_step = (start - end) / steps

    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step

    return itertools.chain(frange(start, end, eps_step), itertools.repeat(end))
