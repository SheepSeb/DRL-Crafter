import torch
from copy import deepcopy

class PrioritizedDQNAgent:
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
    ):
        super().__init__()
        self._env = env
        self._estimator = estimator
        self._target_estimator = deepcopy(estimator).eval()
        self._buffer = buffer
        self._optimizer = optimizer
        self._epsilon = epsilon_schedule
        self._action_num = env.action_space.n
        self._gamma = gamma
        self._update_steps = update_steps
        self._update_target_steps = update_target_steps
        self._warmup_steps = warmup_steps
        self._step_cnt = 0
        assert (
            warmup_steps > self._buffer._batch_size
        ), "You should have at least a batch in the ER."

    def act(self, state):
        with torch.no_grad():
            return self._estimator(state).argmax(dim=1).item()

    def step(self, state):
        if self._step_cnt < self._warmup_steps:
            return int(torch.randint(self._action_num, (1,)).item())

        if next(self._epsilon) < torch.rand(1).item():
            return self.act(state)

        return self._env.get_action()

    def learn(
        self,
        state,
        action,
        reward,
        state_,
        done,
    ):
        self._buffer.push((state.cpu(), action, reward, state_.cpu(), done))

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return

        if self._step_cnt % self._update_steps == 0:
            # sample a batch from experience replay with priorities
            batch = self._buffer.sample()
            
            # batch now includes importance sampling weights and indices
            states, actions, rewards, states_, done, indices, weights = batch

            # perform an update
            td_errors = self._update(states, actions, rewards, states_, done, weights)
            
            # update priorities in the replay buffer
            self._buffer.update_priorities(indices, abs(td_errors.detach().cpu().numpy()))

        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(self._estimator.state_dict())

        self._step_cnt += 1

    def _update(
        self,
        states,
        actions,
        rewards,
        states_,
        done,
        weights
    ):
        # Compute Q(s, * | θ) and Q(s', . | θ^)
        q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(states_)

        # compute Q(s, a) and max_a' Q(s', a')
        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.max(1, keepdim=True)[0]

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
    
    def save(self, path: str):
        torch.save(self._estimator.state_dict(), path)

    def train(self):
        self._estimator.train()

    def eval(self):
        self._estimator.eval()