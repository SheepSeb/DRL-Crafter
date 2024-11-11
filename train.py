import argparse
import pickle
from pathlib import Path

import torch
from torch import nn

from tqdm import tqdm

from src.crafter_wrapper import Env

from src.utils import ReplayMemory, get_epsilon_schedule

from src.networks.net import ConvModel
from src.networks.duel_net import DuelNet

from src.agents.dqn import DQNAgent
from src.agents.double_dqn import DoubleDQNAgent
from src.agents.munchausen_dqn import MunchausenDoubleDQNAgent

class RandomAgent:
    """An example Random Agent"""

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()


def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """ Use the greedy, deterministic policy, not the epsilon-greedy policy you
    might use during training.
    """
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0.0)

        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate"
            + "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(
        f"Observations are of dims ({opt.history_length},84,84),"
        + "with values between 0 and 1."
    )


def main(opt):
    _info(opt)
    
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # opt.device = torch.device("cpu")
    
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    
    # agent = RandomAgent(env.action_space.n)

    if ("duel" in opt.agent):
        print("Network: duel")
        net = DuelNet(opt.history_length, env.action_space.n).to(opt.device)
    else:
        print("Network: conv")
        net = ConvModel(opt.history_length, env.action_space.n).to(opt.device)

    buffer = ReplayMemory(device=opt.device, size=5_000, batch_size=32)

    if ("munchausen" in opt.agent):
        print("Agent: MunchausenDoubleDQN")
        agent = MunchausenDoubleDQNAgent(
                env,
                net,
                buffer,
                torch.optim.Adam(net.parameters(), lr=4e-4, eps=1e-5),
                get_epsilon_schedule(start=0.4, end=0.01, steps=opt.steps * 0.5),
                warmup_steps=opt.steps * 0.025,
                update_steps=4,
                update_target_steps=2_000,
            )
    elif ("double_dqn" in opt.agent):
        print("Agent: DoubleDQN")
        agent = DoubleDQNAgent(
                env,
                net,
                buffer,
                torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5),
                get_epsilon_schedule(start=1.0, end=0.1, steps=opt.steps * 0.5),
                warmup_steps=opt.steps * 0.025,
                update_steps=4,
                update_target_steps=2_000,
            )
    elif ("dqn" in opt.agent):
        print("Agent DQN")
        agent = DQNAgent(
                env,
                net,
                buffer,
                torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5),
                get_epsilon_schedule(start=1.0, end=0.1, steps=opt.steps * 0.5),
                warmup_steps=opt.steps * 0.025,
                update_steps=4,
                update_target_steps=2_000,
            )

    agent.train()

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    pbar = tqdm(total=opt.steps, position=0, leave=True)
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False
            episode_reward = 0

        action = agent.step(obs)
        obs_next, reward, done, info = env.step(action)
        agent.learn(obs, action, reward, obs_next, done)

        step_cnt += 1

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            agent.eval()
            eval(agent, eval_env, step_cnt, opt)
            agent.train()
        
        episode_reward += reward
        obs = obs_next.clone()

        pbar.set_description(
            f"[Episode {ep_cnt}] | Reward {episode_reward:.04f}"
        )
        pbar.update(1)
    
    agent.save(opt.logdir + "/agent.pkl")


def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/check/1")
    parser.add_argument("--agent", default="double_dqn")
    
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        # default=1_000_000,
        default=250_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=25_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
