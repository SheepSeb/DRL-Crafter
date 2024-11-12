import pathlib
from collections import deque

import crafter
from crafter import constants

import numpy as np
import torch
from PIL import Image


class Env:
    def __init__(self, mode, args):
        assert mode in (
            "train",
            "eval",
            "human" # for parsing the human buffer
        ), "`mode` argument can either be `train` or `eval`"
        self.device = args.device

        env = crafter.Env()
        if mode == "train":
            env = crafter.Recorder(
                env,
                pathlib.Path(args.logdir) / "train",
                save_stats=True,
                save_video=False,
                save_episode=False,
            )
        elif mode == "eval":
            env = crafter.Recorder(
                env,
                pathlib.Path(args.logdir) / "eval",
                save_stats=True,
                save_video=False,
                save_episode=False,
            )

        # env = ResizeImage(env)
        env = GrayScale(env)
        self.env = env
        self.action_space = env.action_space
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)

        self.check_valid = (mode == 'train')

    def reset(self):
        for _ in range(self.window):
            # self.state_buffer.append(torch.zeros(84, 84, device=self.device))
            self.state_buffer.append(torch.zeros(64, 64, device=self.device))
        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).div_(255)
        self.state_buffer.append(obs)

        if (self.check_valid):
            reward += self._validate_action(action)

        return torch.stack(list(self.state_buffer), 0).unsqueeze(0), reward, done, info
    
    def _validate_action(self, action):
        ACTIONS_PLACE = {
            7: "stone",
            8: "table",
            9: "furnace",
            10: "plant",
        }

        if (action in ACTIONS_PLACE):
            return 0.0 if (self._check_place(ACTIONS_PLACE[action])) else -0.1

        ACTIONS_MAKE = {
            11: "wood_pickaxe",
            12: "stone_pickaxe",
            13: "iron_pickaxe",
            14: "wood_sword",
            15: "stone_sword",
            16: "iron_sword",
        }

        if (action in ACTIONS_MAKE):
            return 0.0 if (self._check_make(ACTIONS_MAKE[action])) else -0.1
    
        return 0.0

    def _check_place(self, item):
        player = self.env._player

        target = (player.pos[0] + player.facing[0], player.pos[1] + player.facing[1])
        material, _ = player.world[target]

        if player.world[target][1]:
            return False
        
        item_info = constants.place[item]
        if material not in item_info["where"]:
            return False
        
        if np.any(player.inventory[k] < v for k, v in item_info["uses"].items()):
            return False

        return True


    def _check_make(self, item):
        player = self.env._player

        nearby, _ = player.world.nearby(player.pos, 1)
        info = constants.make[item]
        
        if not np.all(util in nearby for util in info["nearby"]):
            return False
        if np.any(player.inventory[k] < v for k, v in info["uses"].items()):
            return False

        return True

    def get_action(self):
        ## check the actions here - https://github.com/danijar/crafter/blob/main/crafter/data.yaml
        actions = [
            (0, 1), # noop
            (1, 5), # move_left
            (2, 5), # move_right
            (3, 5), # move_up
            (4, 5), # move_down
            (5, 10), # do
            (6, 1), # sleep
        ]

        items_place = [
            (7, "stone", 5),
            (8, "table", 10),
            (9, "furnace", 10),
            (10, "plant", 5),
        ]
        
        for idx, item_place, score in items_place:
            if (self._check_place(item_place)):
                actions.append((idx, score))
        
        items_make = [
            (11, "wood_pickaxe", 50),
            (12, "stone_pickaxe", 50),
            (13, "iron_pickaxe", 50),
            (14, "wood_sword", 50),
            (15, "stone_sword", 50),
            (16, "iron_sword", 50),         
        ]

        for idx, item_make, score in items_make:
            if (self._check_make(item_make)):
                actions.append((idx, score))

        actions = np.array(actions)
        return np.random.choice(actions[:, 0], p=actions[:, 1]/(actions[:, 1].sum()))


class GrayScale:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = obs.mean(-1)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = obs.mean(-1)
        return obs


class ResizeImage:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = self._resize(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = self._resize(obs)
        return obs

    def _resize(self, image):
        image = Image.fromarray(image)
        image = image.resize((84, 84), Image.NEAREST)
        image = np.array(image)
        return image
