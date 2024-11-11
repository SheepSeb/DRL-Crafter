import torch
import torch.nn.functional as F

from src.agents.double_dqn import DoubleDQNAgent

class MunchausenDoubleDQNAgent(DoubleDQNAgent):
    def __init__(
        self,
        env,
        estimator,
        buffer,
        optimizer,
        epsilon_schedule,
        gamma=0.99,
        update_steps=2,
        update_target_steps=2000,
        warmup_steps=100,
        alpha=0.9,  # Munchausen scaling factor
        tau=0.03,   # Temperature parameter for softmax
        lo=-1,      # Lower bound for log-policy
    ):
        super().__init__(
            env=env,
            estimator=estimator,
            buffer=buffer,
            optimizer=optimizer,
            epsilon_schedule=epsilon_schedule,
            gamma=gamma,
            update_steps=update_steps,
            update_target_steps=update_target_steps,
            warmup_steps=warmup_steps,
        )
        self._alpha = alpha
        self._tau = tau
        self._lo = lo

    def _compute_log_policy(self, q_values):
        """Compute the log policy using softmax with temperature tau."""
        return torch.log_softmax(q_values / self._tau, dim=1)

    def _update(
        self,
        states,
        actions,
        rewards,
        states_,
        done,
    ):
        # Compute Q-values for current and next states
        q_values = self._estimator(states)
        
        with torch.no_grad():
            # Get next actions using online network (Double DQN)
            next_actions = self._estimator(states_).argmax(1, keepdim=True)
            # Get next Q-values using target network
            next_q_values = self._target_estimator(states_)
            
            # Compute log-policy for current and next states
            log_pi = self._compute_log_policy(q_values)
            next_log_pi = self._compute_log_policy(next_q_values)
            
            # Clip log-policy to prevent numerical instability
            log_pi = torch.clamp(log_pi, min=self._lo)
            next_log_pi = torch.clamp(next_log_pi, min=self._lo)
            
            # Compute Munchausen term for current reward
            munchausen_term = self._alpha * self._tau * log_pi.gather(1, actions)
            
            # Compute modified reward
            modified_rewards = rewards + munchausen_term
            
            # Compute next state value with Munchausen term
            next_q = next_q_values.gather(1, next_actions)
            next_munchausen_term = self._alpha * self._tau * next_log_pi.gather(1, next_actions)
            next_value = next_q + next_munchausen_term
            
            # Compute target Q-value
            target_qsa = modified_rewards + self._gamma * next_value * (1 - done.float())
        
        # Compute current Q-value
        qsa = q_values.gather(1, actions)
        
        # Compute loss and update
        loss = (qsa - target_qsa).pow(2).mean()
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()