"""
TD(0) & SARSA algorithms for the Blackjack environment in Gymnasium.

The house strategy is stick at 16 or larger.
"""
from gymnasium.envs.toy_text.blackjack import BlackjackEnv, sum_hand
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class CustomBlackjackEnv(BlackjackEnv):
    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)
    
    def dealer_policy(self, hand):
        # The house strategy is stick at 16 or larger, if sum_hand(hand) < 16 then hit 
        return sum_hand(hand) < 16

# Create environment
env = CustomBlackjackEnv(sab=False, natural=False)
num_episodes = 1000000  # Number of episodes for training
alpha = 0.01  # Learning rate
num_states = env.observation_space[0].n
num_actions = env.action_space.n  # Actions: Stick (0), Hit (1)

def train(learning_function, action_function):
    # --- Training loop ---
    for _ in tqdm(range(num_episodes)):
        state, _ = env.reset()

        done = False

        while not done:
            # We compute the next action again since the Q function may have changed
            action = action_function(state[0])
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # modify the reward structure to be 0, 1 for counting only gambler winnings.
            reward = reward > 0
            # learning_function
            next_action = action_function(next_state[0])
            learning_function(state[0], action, reward, next_state[0], next_action, done)
            # Move to next state
            state = next_state
            

def td0_learning_function(state, _, reward, next_state, __, done):
    v_vec[state] += alpha * (reward + v_vec[next_state] * (not done)- v_vec[state])
    
def sarsa_learning_function(state, action, reward, next_state, next_action, done):
    Q[state, action] += alpha * (reward + Q[next_state, next_action] * (not done) - Q[state, action])

def Q_action(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(Q[state])        # Exploit

""" TD(0) Learning """
v_vec = np.zeros((num_states))  # State-value function
train(td0_learning_function, lambda s: s < 18)  # Simple policy: hit if state < 18, else stick

""" SARSA Learning """
Q = np.zeros((num_states, num_actions))  # Action-value function
train(sarsa_learning_function, Q_action)

plt.figure(figsize=(10, 6))
plt.imshow(Q.T, origin='lower')
plt.colorbar(label='Q Value')
plt.savefig(f'ex4_q3_blackjack_sarsa_Q_{num_episodes}_episodes.png')

plt.clf()
plt.bar(np.arange(num_states), np.argmax(Q, axis=1))
plt.ylabel('Best Action (0=Stick, 1=Hit)')
plt.xlabel('Player Sum')
plt.title(f'Optimal Policy after SARSA Learning - {num_episodes} Episodes')
plt.savefig(f'ex4_q3_blackjack_sarsa_action_{num_episodes}_episodes.png')

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(v_vec)), v_vec, "o--", label='TD(0) Value Estimates')
plt.plot(np.arange(len(v_vec)), np.max(Q, axis=1), "x--", label='SARSA Value Estimates')
plt.legend()
plt.xlabel('Player Sum')
plt.ylabel('Probability of winning')
plt.title(f'State-Value Function Estimates - {num_episodes} Episodes')
plt.grid(which="both")
plt.savefig(f'ex4_q3_blackjack_{num_episodes}_episodes.png')
