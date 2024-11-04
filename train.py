import argparse
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from src.crafter_wrapper import Env
from src.agents.random_agent import RandomAgent
from src.agents.reinforce_agent import ReinforceAgent, Policy
from src.agents.actor_critic_agent import A2C_Agent, ActorCriticPolicy

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
        episodic_returns.append(0)
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
    #opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = torch.device("cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)

    if opt.agent_type == "random":
        agent = RandomAgent(env.action_space.n)
    elif opt.agent_type == "reinforce":
        policy = Policy(84 * 84 * opt.history_length, env.action_space.n)
        agent = ReinforceAgent(policy, 0.99, torch.optim.Adam(policy.parameters(), lr=1e-2))
    elif opt.agent_type == "a2c":
        policy = ActorCriticPolicy(84 * 84 * opt.history_length, env.action_space.n)
        agent = A2C_Agent(policy, 0.99, torch.optim.Adam(policy.parameters(), lr=5e-3, eps=1e-05), nsteps=20)
    else:
        raise ValueError(f"Unknown agent type: {opt.agent_type}")

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    pbar = tqdm(total=opt.steps, desc="Training")

    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False

        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        step_cnt += 1
        pbar.update(1)
        pbar.set_postfix({"episode": ep_cnt, "reward": reward})

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)
    
    pbar.close()

def get_options():
    """ Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", type=str, default="reinforce", choices=["random", "reinforce","a2c"], help="Type of agent to use.")
    parser.add_argument("--logdir", default="logdir/random_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
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
        default=100_000,
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
