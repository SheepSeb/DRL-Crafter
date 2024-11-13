import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.agents.reinforce_agent import Reinforce, Policy

class ActorCriticPolicy(nn.Module):
    def __init__(self, state_sz, action_num):
        super().__init__()
        # Modify the architecture to use a CNN-LSTM
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=100, hidden_size=512, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32768, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.policy = nn.Linear(512, action_num)
        self.head_value = nn.Linear(512, 1)

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
        pi = Categorical(F.softmax(self.policy(x),dim=-1))
        value = self.head_value(x)
        return pi, value

class A2C(Reinforce):
    def __init__(self, *args, nsteps=5, **kwargs):
        super().__init__(*args, **kwargs)
        self._nsteps = nsteps
        self._beta = 0.01       # beta term in entropy regularization
        self._values = []       # keeps episodic/nstep value estimates
        self._entropies = []    # keeps episodic/nstep policy entropies
        self._step_cnt = 0

    def act(self, state, eval=False):
        pi, value = self._policy(state)
        action = pi.sample()
        if not eval:
            self._log_probs.append(pi.log_prob(action))
            self._values.append(value)
            self._entropies.append(pi.entropy())
        return action.item()
    
    def learn(self, state, action, reward, state_, done):
        self._rewards.append(reward)

        if done or (self._step_cnt % (self._nsteps - 1) == 0 and self._step_cnt != 0):
            self._update_policy(done, state_)

        self._step_cnt = 0 if done else self._step_cnt + 1
    
    def _compute_returns(self, done, state_):
        returns = []
        R = self._policy(state_)[1].detach() * (1 - done)
        for r in self._rewards[::-1]:
            R = r + self._gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self._fp32_err)
        return returns  # ఠ_ఠ
 
    def _update_policy(self, done, state_):
        returns = self._compute_returns(done, state_)

        values = torch.cat(self._values).squeeze(0)
        log_probs = torch.stack(self._log_probs).squeeze()
        entropy = torch.stack(self._entropies).squeeze()
        advantage = returns.to(values.device) - values

        policy_loss = (-log_probs * advantage.detach()).sum()
        returns = returns.to(values.device)
        critic_loss = F.smooth_l1_loss(values, returns)

        self._optimizer.zero_grad()
        (policy_loss + critic_loss - self._beta * entropy.mean()).backward()
        self._optimizer.step()

        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._entropies.clear()