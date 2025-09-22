import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
we solve CartPole problem, see:
https://www.gymlibrary.dev/environments/classic_control/cart_pole/
"""
NO_SEED = -1

def apply_episode(env, w, num_steps):
	accumulated_reward = 0
	if SEED == NO_SEED:
		observation, _ = env.reset()
	else:
		observation, _ = env.reset(seed=SEED)
		
	for _ in range(num_steps):
	   action = np.heaviside(np.dot(w, observation), 1).astype(int)
	   
	   observation, reward, terminated, truncated, _ = env.step(action)
	   
	   # If the episode has ended then we can reset to start a new episode
	   accumulated_reward += reward
	   if terminated or truncated:
	   	break
	return accumulated_reward

def random_search(env, num_steps, num_epochs, exit_score):
	if SEED == NO_SEED:
		observation, _ = env.reset()
	else:
		observation, _ = env.reset(seed=SEED)
		
	n = observation.shape
	maximal_reward = -np.inf
	num_epochs_required = 0
	for idx_epoch in range(num_epochs):
		w = np.random.choice((-1, 1), size=n)
		reward = apply_episode(env, w, num_steps)
		
		if reward > maximal_reward:
			maximal_reward = reward
			w_opt = w
			
			if maximal_reward >= exit_score:
				# if exit_score==inf do not stop
				num_epochs_required = idx_epoch + 1
				break
	return w_opt, maximal_reward, num_epochs_required
			
def evaluate_num_iterations(env, num_steps, num_epochs, exit_score, num_evaluations):
	num_epochs_required = np.zeros((num_evaluations))
	max_reward = np.zeros((num_evaluations))
	for idx_evaluation in tqdm(range(num_evaluations)):
		_, max_reward[idx_evaluation], num_epochs_required[idx_evaluation] = random_search(env, num_steps, num_epochs, exit_score)
	print(f"{np.sum(max_reward == exit_score)} / {num_evaluations} reached maximal score")
	return num_epochs_required
	
	
if __name__ == '__main__':
	env = gym.make("CartPole-v0")
	SEED = -1 # if -1 do not set seed
	
	res = evaluate_num_iterations(env, num_steps=200, num_epochs=10000, exit_score=200, num_evaluations=1000)
	env.close()
	
	plt.hist(res)
	plt.title(f"mean(num iteration): {np.mean(res):.2f}")
	plt.xlabel("num iterations until optimal")
	plt.savefig("ex1_q6_hist.jpg")
	
	