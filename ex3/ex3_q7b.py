import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt

# --- Define Q-Network ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, action_size, bias=False)
    def forward(self, state):
        return self.fc1(state)

# --- Convert state index → one-hot vector ---
def state2hot(state_index, state_size):
    s = torch.zeros((1, state_size))
    s[0, state_index] = 1.0
    return s

# --- Environment setup ---
env = gym.make('FrozenLake-v1', is_slippery=True)
state_size = env.observation_space.n
action_size = env.action_space.n

# --- Initialize networks ---
q_network = QNetwork(state_size, action_size)
target_network = copy.deepcopy(q_network)

# --- Hyperparameters ---
loss_fn = nn.MSELoss()
optimizer = optim.SGD(q_network.parameters(), lr=0.1)
gamma = 0.99
epsilon = 0.1
num_episodes = 20000
TARGET_UPDATE_FREQUENCY = 50  # update target after N episodes

rList = []

# --- Training loop ---
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for step in range(99):
        s_hot = state2hot(state, state_size)

        # ε-greedy policy
        with torch.no_grad():
            q_values = q_network(s_hot)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q_values).item()

        # Take action
        next_state, reward, done, truncated, _ = env.step(action)
        next_s_hot = state2hot(next_state, state_size)

        # Compute Bellman target using target network
        with torch.no_grad():
            q_next = target_network(next_s_hot)
            max_q_next = torch.max(q_next)
            target_value = reward + gamma * max_q_next * (0 if done else 1)

        # Predicted Q(s,a)
        q_pred = q_network(s_hot)[0, action]

        # Compute loss and optimize
        loss = loss_fn(q_pred, torch.tensor(target_value))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        state = next_state
        if done or truncated:
            break

    rList.append(total_reward)

    # Decay epsilon
    epsilon = max(0.01, 1. / ((episode / 50) + 10))

    # Update target network every N episodes
    if (episode + 1) % TARGET_UPDATE_FREQUENCY == 0:
        target_network.load_state_dict(q_network.state_dict())
        print(f"✅ Target network updated at episode {episode + 1}")

print("Average score over time:", np.mean(rList))
# 0.65735

# --- 📈 Plot reward and moving average ---
window = 50  # moving average window
movmean = np.convolve(rList, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10,5))
plt.plot(rList, color='lightblue', label='Reward (per episode)', alpha=0.5)
plt.plot(range(window-1, len(rList)), movmean, color='blue', linewidth=2, label=f'Moving average ({window})')
plt.title('FrozenLake-v1: Q-Network Learning Progress')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
