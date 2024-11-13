import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, state_sz, action_num):
        super(Policy, self).__init__()
        # Modify the architecture to use a CNN-LSTM
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=100, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32768, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(512, action_num)

    def forward(self, x):
        """ Returns a torch.distributions.Categorical object. Check the docs!
            https://pytorch.org/docs/stable/distributions.html
        """
        # The input vector is a tensor of shape (batch_size, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x, _ = self.lstm(x.unsqueeze(1))
        # Linearize the x tensor
        x = x.flatten()
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        scores = self.affine2(x)
        return Categorical(F.softmax(scores, dim=0))

class Reinforce:
    def __init__(self, policy, gamma, optimizer):
        self._policy = policy
        self._gamma = gamma
        self._optimizer = optimizer
        self._fp32_err = 2e-07  # used to avoid division by 0
        self._log_probs = []
        self._rewards = []
    
    def act(self, state, eval=False):
        """ Receives a torch tensor for the current state and returns
        and action (integer).
        """
        pi = self._policy(state)
        action = pi.sample()
        if not eval:
            self._log_probs.append(pi.log_prob(action))
        return action.item()
    
    def learn(self, state, action, reward, state_, done):
        self._rewards.append(reward)

        if done:
            self._update_policy()

    def _compute_returns(self):
        """ Use `self._rewards` to compute a vector of discounted expected
        returns.

            This function should return a tensor the size of the rewards.
        """
        R, returns = 0, []
        for r in self._rewards[::-1]:
            R = r + self._gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # Here we normalize the returns. This is a rather theoretically unsound
        # trick but it helps with the speed of convergence in this environment.
        returns = (returns - returns.mean()) / (returns.std() + self._fp32_err)

        return returns
 
    def _update_policy(self):
        returns = self._compute_returns()
        log_probs = torch.stack(self._log_probs).squeeze()
        policy_loss = (-log_probs * returns.to(log_probs.device)).sum()

        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()

        del self._rewards[:]
        del self._log_probs[:]