"""
we solve FrozenLake problem:
	
The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in:
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
Action indices [0, 1, 2, 3] correspond to West, South, East and North.
env.P[state][action] is a list of tuples (probability, nextstate, reward, terminated)
E.g.:
P[0][0] = [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False)]
nS - number of states
nA - number of actions
"""
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)

def convert_P(env):
	# for more complicated case one can use sparse matrices, e.g. lil_array
	# P(s_new|s, a) - indices: (s_old, action, s_new)
	nA = env.action_space.n
	nS = env.observation_space.n
	P = np.zeros((nS, nA, nS))
	R = np.zeros((nS, nA, nS))
	for idx_old_state in range(nS):
		for idx_action in range(nA):
			possible_observations = env.P[idx_old_state][idx_action]
			for observation in possible_observations:
				P[idx_old_state][idx_action][observation[1]] += observation[0]
				R[idx_old_state][idx_action][observation[1]] = observation[2]
	return P, R
	

def value_iteration(env, gamma, num_iter, verbose=True):
    """
    Inputs:
        env
        gamma: discount factor
        num_iter: number of iterations	
    Outputs:
        (value_functions, policies)

    size(value_functions) == (num_iter+1, nS) and size(policies) == (num_iter, nS)
    """
    nS = env.observation_space.n
    V = np.zeros((num_iter+1, nS))
    pi_s = np.zeros((num_iter, nS), dtype=int)
    P, R = convert_P(env)
    if verbose:
    	print("Iteration | max|V-Vprev| | # chg actions | V[0]")
    	print("----------+--------------+---------------+---------")
	   
    for idx_iter in range(num_iter):
    	A = P * (R + gamma * V[idx_iter].reshape((1, 1, -1)))
    
    	Tv = np.sum(A, axis=2)
    	pi_s[idx_iter] = np.argmax(Tv, axis=1).astype(int)
    	
    	V[idx_iter + 1] = np.take_along_axis(Tv, pi_s[idx_iter].reshape((-1, 1)), axis=1).squeeze()
    	
    	max_diff = np.abs(V[idx_iter + 1] - V[idx_iter]).max()
    	nChgActions = np.sum(pi_s[idx_iter] != pi_s[idx_iter - 1])
    	if verbose:
    		print("%4i      | %6.5f      | %4s          | %5.3f"%(idx_iter, max_diff, nChgActions, V[idx_iter][0]))
    return V, pi_s

def plot_policy(Vs_VI, pis_VI, axis_length):
    for idx_iter, (V, pi) in enumerate(zip(Vs_VI, pis_VI)):
    	plt.figure(figsize=(3,3))
    	plt.imshow(V.reshape(axis_length, axis_length), cmap='gray', interpolation='none', clim=(0,1))
    	ax = plt.gca()
    	ax.set_xticks(np.arange(axis_length)-.5)
    	ax.set_yticks(np.arange(axis_length)-.5)
    	ax.set_xticklabels([])
    	ax.set_yticklabels([])
    	plt.colorbar()
    	Y, X = np.mgrid[0:axis_length, 0:axis_length]
    	a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(0, 1)}
    	Pi = pi.reshape(axis_length,axis_length)
    	for y in range(axis_length):
    	       for x in range(axis_length):
    	       	a = Pi[y, x]
    	       	u, v = a2uv[a]
    	       	plt.arrow(x, y,u*.3, -v*.3, color='m', head_width=0.1, head_length=0.1)
    	       	plt.text(x, y, str(env.desc[y,x].item().decode()), color='g', size=12, verticalalignment='center', horizontalalignment='center', fontweight='bold')
    	       	plt.grid(color='b', lw=2, ls='-')
    	plt.title(f"iteration {idx_iter}")
    	plt.savefig(f"iteration_{idx_iter}.jpg")
    	plt.close()

def plot_values(Vs_VI):
	plt.figure()
	plt.plot(Vs_VI)
	plt.xlabel("# iteration")
	plt.ylabel("state value")
	plt.legend([f"state {k}" for k in range(Vs_VI.shape[1])])
	plt.grid()
	plt.savefig("state_values.jpg")
	
if __name__ == '__main__':
	env = gym.make("FrozenLake-v1").unwrapped
	Vs_VI, pis_VI = value_iteration(env, gamma=0.95, num_iter=20)
	plot_policy(Vs_VI, pis_VI, 4)
	plot_values(Vs_VI)
	
	env.close()
 
