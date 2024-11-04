import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, state_sz, action_num):
        super().__init__()
        self.affine1 = nn.Linear(state_sz, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, action_num)
    
    def forward(self, x):
        # Linearize x to a 1D tensor with the shape 1 x state_sz
        x = x.reshape(1, -1)
        x = self.affine2(F.relu(self.dropout(self.affine1(x))))
        return Categorical(F.softmax(x, dim=1))
    

class ReinforceAgent:
    def __init__(self, policy, gamma, optimizer):
        self._policy = policy
        self._gamma = gamma
        self._optimizer = optimizer
        self._fp32_err = 2e-07
        self._log_probs = []
        self._rewards = []

    def act(self, state):
        pi = self._policy(state)
        action = pi.sample()
        self._log_probs.append(pi.log_prob(action))
        return action.item()
    
    def learn(self, state, action, reward, state_, done):
        self._rewards.append(reward)
        if done:
            self._update()

    def _compute_returns(self):
        R, returns = 0, []
        for r in self._rewards[::-1]:
            R = r + self._gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        returns = (returns - returns.mean()) / (returns.std() + self._fp32_err)
        return returns
    
    def _update_policy(self):
        returns = self._compute_returns()
        log_probs = torch.cat(self._log_probs)
        policy_loss = (-log_probs * returns.to(log_probs.device)).sum()

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        del self._rewards[:]
        del self._log_probs[:]