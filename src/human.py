import argparse
from pathlib import Path

import numpy as np

from src.crafter_wrapper import Env

## env inside the Env() in crafter_wrappper
class HumanEnv():
    def __init__(self, records):
        self._records = records
        self.files = sorted(list(Path(records).glob("*/*.npz")))
        self.file_idx, self.step_idx = 0, 0
    
    def reset(self):
        game = np.load(self.files[self.file_idx], allow_pickle=True)
        self.file_idx += 1

        self.step_idx = 0

        self.image = game["image"]
        self.action = game["action"]
        self.reward = game["reward"]
        self.done = game["done"]

        # return the first observation on reset
        res = self.image[0]
        return res.mean(-1)
    
    def step(self, action):
        self.step_idx += 1

        res = (
            self.image[self.step_idx].mean(-1),
            self.reward[self.step_idx],
            self.done[self.step_idx],
            None # no info
        )
        
        return res
    
    def get_action(self):
        return self.action[self.step_idx + 1]

def retrieve_human_buffer(records, opt):
    opt.device = 'cpu' # store all the states on cpu
    crafter_env = Env(mode="human", args=opt)
    
    human_env = HumanEnv(records)

    crafter_env.env = human_env

    buffer = []

    obs = crafter_env.reset()
    while True:
        action = crafter_env.env.get_action()

        obs_, reward, done, _ = crafter_env.step(action)

        buffer.append((
            obs.cpu(),
            action,
            reward,
            obs_.cpu(),
            done
        ))

        if (done and crafter_env.env.file_idx == len(crafter_env.env.files)):
            # parsed all the games
            break

        if (done):
            obs = crafter_env.reset()
    
    return buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    opt = parser.parse_args()

    opt.device = 'cpu'

    records = "_human"
    
    buffer = retrieve_human_buffer(records, opt=opt)
    # x = 10
