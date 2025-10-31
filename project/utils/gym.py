"""
This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import gymnasium as gym
from gymnasium import wrappers
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.monitor import Monitor

from utils.seed import set_global_seeds
from utils.atari_wrapper import wrap_deepmind, wrap_deepmind_ram

# def get_env(task, seed):
#     env_id = task.env_id

#     env = gym.make(env_id)

#     set_global_seeds(seed)
#     env.seed(seed)

#     expt_dir = 'tmp/gym-results'
#     env = wrappers.Monitor(env, expt_dir, force=True)
#     env = wrap_deepmind(env)

#     return env

# def get_ram_env(env, seed):
#     set_global_seeds(seed)
#     env.seed(seed)

#     expt_dir = '/tmp/gym-results'
#     env = wrappers.Monitor(env, expt_dir, force=True)
#     env = wrap_deepmind_ram(env)

#     return env


def get_record_video(env):
    return Monitor(env)
