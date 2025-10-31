import gymnasium as gym
import torch.optim as optim
import ale_py

from dqn_model import DQN
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_record_video
from utils.schedule import LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 5000  # 50000
LEARNING_STARTS = 5000  # 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01

gym.register_envs(ale_py)


def main(env, num_timesteps):
    def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_record_video(env).get_total_steps() >= num_timesteps
        # return False  # Disable stopping criterion for testing purposes

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    dqn_learing(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGER_UPDATE_FREQ,
        num_timesteps=num_timesteps,
    )


if __name__ == "__main__":
    # Get Atari games.
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    # env = gym.make("ALE/Pong-v5", obs_type="grayscale", render_mode="rgb_array")

    # max_timesteps = env.spec.max_episode_steps

    main(env, num_timesteps=1e6)
