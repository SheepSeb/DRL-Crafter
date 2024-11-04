import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.agents.reinforce_agent import ReinforceAgent, Policy

class ActorCriticPolicy(nn.Module):
    def __init__(self, state_sz, action_num, hidden_state = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_sz, hidden_state)
        self.policy = nn.Linear(hidden_state, action_num)
        self.value = nn.Linear(hidden_state, 1)

    def forward(self, x):
        x = x.reshape(1, -1)
        x = F.relu(self.fc1(x))
        pi = Categorical(F.softmax(self.policy(x), dim=-1))
        value = self.value(x)
        return pi, value
    

class A2C_Agent(ReinforceAgent):
    def __init__(self, *args, nsteps = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self._nsteps = nsteps
        self._beta = 0.01
        self._values = []
        self._entropies = []
        self._step_cnt = 0

    
    def act(self, state):
        pi, value = self._policy(state)
        action = pi.sample()
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
        R = self._policy(self._state_)[1].detach() * (1-done)
        for r in self._rewards[::-1]:
            R = r + self._gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self._fp32_err)
        return returns
    
    def _update_policy(self, done, state_):
        returns = self._compute_returns(done, state_)

        values = torch.cat(self._values).squeeze(1)
        log_probs = torch.cat(self._log_probs)
        entropy = torch.cat(self._entropies)
        advantage = returns.to(values.device) - values

        policy_loss = (-log_probs * advantage.detach()).sum()
        critic_loss = F.smooth_l1_loss(values, returns)

        self._optimizer.zero_grad()
        (policy_loss + critic_loss - self._beta * entropy.mean()).backward()
        self._optimizer.step()

        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._entropies.clear()