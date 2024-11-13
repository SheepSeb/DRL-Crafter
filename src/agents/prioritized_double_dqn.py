import torch

from src.agents.prioritized_dqn import PrioritizedDQNAgent

class PrioritizedDoubleDQNAgent(PrioritizedDQNAgent):
    def _update(
        self,
        states,
        actions,
        rewards,
        states_,
        done,
        weights
    ):
        # compute the DeepQNetwork update. Carefull not to include the
        # target network in the computational graph.

        # Compute Q(s, * | θ) and Q(s', . | θ^)
        with torch.no_grad():
            actions_ = self._estimator(states_).argmax(1, keepdim=True)
            q_values_ = self._target_estimator(states_)
        q_values = self._estimator(states)

        # compute Q(s, a)
        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.gather(1, actions_)

        # compute target Q(s', a')
        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        # compute TD errors for updating priorities
        td_errors = qsa - target_qsa

        # compute the weighted loss using importance sampling weights
        loss = (weights * td_errors.pow(2)).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        return td_errors  # return TD errors for updating priorities
