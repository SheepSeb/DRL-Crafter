import torch

from collections import deque

import random

import itertools

class ReplayMemory:
    """Cyclic buffer that stores the transitions of the game on CPU RAM."""

    def __init__(self, device, size=1000, batch_size=32):
        self._buffer: deque = deque(maxlen=size)
        self._batch_size = batch_size
        self._device = device

    def push(self, transition):
        """Store the transition in the buffer

        The first element of the transition is the current state with the shape
        (1, ...), the second element will be the action that the agent took,
        the third will be the reward, the fourth will be the next state with
        the shame shape as the current state, and the last will be a boolen
        that will tell if the game is done.

        The function will move the tensors to cpu.
        """
        s, a, r, s_, d = transition
        self._buffer.append((s.cpu(), a, r, s_.cpu(), d))

    def sample(self):
        """Sample from self._buffer

        Should return a tuple of tensors of size:
        (
            states:     N , ...,
            actions:    N * 1, (torch.int64)
            rewards:    N * 1, (torch.float32)
            states_:    N * ...,
            done:       N * 1, (torch.uint8)
        )

        where N is the batch_size.
        """
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
