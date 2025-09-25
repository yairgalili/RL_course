import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def moving_average(data, window_size):
    """
    Calculates the moving average of a 1D NumPy array.

    Args:
        data (np.ndarray): The input 1D NumPy array.
        window_size (int): The size of the moving average window.

    Returns:
        np.ndarray: The array containing the moving average.
    """
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, 'same')

# Load environment
env = gym.make('FrozenLake-v1')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
gamma = .95
epsilon = 0.1
num_episodes = 200000
verbose = False
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
for i in tqdm(range(num_episodes)):
    #Reset environment and get first new observation
    s, info = env.reset()
    rAll = 0 # Total reward during current episode
    d = False
    t = 0
    #The Q-Table learning algorithm
    while t < 1000:
        t+=1
        # 1. Choose an action by greedily (with noise) picking from Q table
        if np.random.rand(1) < min(epsilon, epsilon / ((i+1)/1000)):
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])

        # 2. Get new state and reward from environment
        s_new, reward, terminated, truncated, _ = env.step(a)
        # 3. Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (reward + gamma * np.max(Q[s_new]) - Q[s, a]) 
        # 4. Update total reward
        rAll += reward
        # 5. Update episode if we reached the Goal State
        s = s_new
        if terminated or truncated:
            if verbose:
                print(f"Episode finished after step {t+1}. Final observation: {s_new}")
            break    
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
# Score over time: 0.632715
print("Final Q-Table Values")
print(Q)
"""
Final Q-Table Values
[[2.64417953e-01 1.96790325e-02 2.15382322e-02 8.40772625e-02]
 [1.48612486e-05 1.03911238e-05 1.54290115e-03 1.97830187e-01]
 [4.39879990e-03 7.91894557e-03 7.94337084e-04 2.33153816e-01]
 [9.60031598e-05 1.20060235e-05 1.89337193e-05 7.68731892e-02]
 [3.85992201e-01 1.54797791e-02 1.58638451e-02 2.00582868e-02]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [7.53705292e-06 2.65099499e-08 2.41877728e-01 1.04559073e-15]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [9.31115026e-03 1.38455840e-02 1.52624237e-03 3.10738969e-01]
 [1.68506302e-03 5.79342924e-01 1.58165488e-03 1.20185548e-02]
 [7.70169650e-01 1.28625649e-03 1.24729806e-03 1.71364789e-04]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]
 [5.48249935e-04 3.26320438e-02 6.72875430e-01 3.00027549e-03]
 [8.72242880e-02 9.39646070e-01 9.08458476e-02 9.70613827e-02]
 [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00]]
"""

plt.plot(range(len(rList)), moving_average(rList, 100), color="blue")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per episode")
plt.savefig("reward_per_episode.jpg")

