"""
This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""

import copy
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gymnasium as gym

import torch
from skimage.transform import resize

import torch.autograd as autograd
from tqdm import tqdm

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_record_video

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda" if USE_CUDA else "cpu")


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


def resize_transform(image_array):
    return resize(image_array, (96, 96), anti_aliasing=True)


Statistic = {"mean_episode_rewards": [], "best_mean_episode_rewards": []}


def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    stopping_criterion=None,
    replay_buffer_size=50000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    num_timesteps=1e6,
):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)

    monitor_env = get_record_video(env)
    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        _, _, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)

        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            with torch.no_grad():
                return model(obs).data.max(1)[1]
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    # Initialize target q function and q function, i.e. build the model.
    ######

    # YOUR CODE HERE
    # --- Initialize networks ---
    Q = q_func(input_arg, num_actions).to(device)
    target_network = copy.deepcopy(Q)

    ######

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float("nan")
    best_mean_episode_reward = -float("inf")
    last_obs = resize_transform(env.reset()[0])
    LOG_EVERY_N_STEPS = 10000

    for t in tqdm(range(int(num_timesteps))):
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)
        #####

        # YOUR CODE HERE
        idx = replay_buffer.store_frame(last_obs)
        recent_obs = replay_buffer.encode_recent_observation()
        action = select_epilson_greedy_action(Q, recent_obs, t).item()
        obs, reward, done, truncated, _ = env.step(action)
        replay_buffer.store_effect(idx, action, reward, done)
        if done or truncated:
            last_obs = resize_transform(env.reset()[0])
        else:
            last_obs = resize_transform(obs)
        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (
            t > learning_starts
            and t % learning_freq == 0
            and replay_buffer.can_sample(batch_size)
        ):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # Note: Move the variables to the GPU if avialable
            # 3.b: fill in your own code to compute the Bellman error. This requires
            # evaluating the current and next Q-values and constructing the corresponding error.
            # Note: don't forget to clip the error between [-1,1], multiply is by -1 (since pytorch minimizes) and
            #       maskout post terminal status Q-values (see ReplayBuffer code).
            # 3.c: train the model. To do this, use the bellman error you calculated perviously.
            # Pytorch will differentiate this error for you, to backward the error use the following API:
            #       current.backward(d_error.data.unsqueeze(1))
            # Where "current" is the variable holding current Q Values and d_error is the clipped bellman error.
            # Your code should produce one scalar-valued tensor.
            # Note: don't forget to call optimizer.zero_grad() before the backward call and
            #       optimizer.step() after the backward call.
            # 3.d: periodically update the target network by loading the current Q network weights into the
            #      target_Q network. see state_dict() and load_state_dict() methods.
            #      you should update every target_update_freq steps, and you may find the
            #      variable num_param_updates useful for this (it was initialized to 0)
            #####

            # YOUR CODE HERE

            # 3.a: Sample a batch from the replay buffer
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = (
                replay_buffer.sample(batch_size)
            )
            obs_batch = torch.tensor(obs_batch, device=device, dtype=torch.float32)
            act_batch = torch.tensor(act_batch, device=device, dtype=torch.long)
            rew_batch = torch.tensor(rew_batch, device=device, dtype=torch.float32)
            next_obs_batch = torch.tensor(
                next_obs_batch, device=device, dtype=torch.float32
            )
            done_mask = torch.tensor(done_mask, device=device, dtype=torch.bool)

            # 3.b: Compute Bellman error
            # Get current Q-values for the actions taken
            current_q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze(1)

            # Compute next Q-values using target network
            with torch.no_grad():
                next_q_values = target_network(next_obs_batch).max(1)[0]
                # Mask out Q-values for terminal states (done_mask is True when episode ended)
                next_q_values[done_mask] = 0.0
                target_q_values = rew_batch + gamma * next_q_values

            # Compute Bellman error (TD error)
            bellman_error = target_q_values - current_q_values

            # Clip the error to [-1, 1]
            clipped_bellman_error = torch.clamp(bellman_error, -1.0, 1.0)

            # Since PyTorch minimizes loss, and we want to minimize squared error,
            # but the instruction says to multiply by -1 and use custom backward,
            # we follow the given API: use -clipped error as "d_error"
            d_error = -clipped_bellman_error

            # 3.c: Train the model
            optimizer.zero_grad()
            # Backward pass using the specified API
            current_q_values.backward(d_error)
            optimizer.step()

            # 3.d: Periodically update target network
            num_param_updates += 1
            if num_param_updates % target_update_freq == 0:
                target_network.load_state_dict(Q.state_dict())
        #####

        ### 4. Log progress and keep track of statistics
        episode_rewards = monitor_env.get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(
                best_mean_episode_reward, mean_episode_reward
            )

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # Dump statistics to pickle
            with open("statistics.pkl", "wb") as f:
                pickle.dump(Statistic, f)
                print("Saved to %s" % "statistics.pkl")
