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
	

def compute_vpi(pi, P, R, gamma):
    # use pi[state] to access the action that's prescribed by this policy
    # we solve:
    # v = sum_s'(P[s, pi[s], :] * R[s, pi[s], :]) + P[s, pi[s], :] * gamma * v
    s = np.arange(pi.size)
    b = np.sum(P[s, pi[s], :] * R[s, pi[s], :], axis=-1)
    A = np.eye(b.size) - gamma * P[s, pi[s], :]
    V = np.linalg.solve(A, b)
    return V

def compute_qpi(vpi, P, R, gamma):
    Qpi = np.sum(P * (R + gamma * vpi.reshape((1, 1, -1))), axis=2)
    return Qpi

def policy_iteration(env, gamma, num_iter, verbose=True):
    """
    Inputs:
        env
        gamma: discount factor
        num_iter: number of iterations	
    Outputs:
        (value_functions, policies)

    size(value_functions) == (num_iter, nS) and size(policies) == (num_iter + 1, nS)
    """
    nS = env.observation_space.n
    V = np.zeros((num_iter, nS))
    pi_s = np.zeros((num_iter + 1, nS), dtype=int)
    P, R = convert_P(env)
    if verbose:
    	print("Iteration | # chg actions | V[0]")
    	print("----------+---------------+---------")
	   
    for idx_iter in range(num_iter):
    	V[idx_iter] = compute_vpi(pi_s[idx_iter], P, R, gamma)
    	
    	Q = compute_qpi(V[idx_iter], P, R, gamma)    
    	
    	pi_s[idx_iter + 1] = np.argmax(Q, axis=1).astype(int)
    	
    	if verbose:
    		print("%4i      | %6i        | %6.5f"%(idx_iter, (pi_s[idx_iter + 1] != pi_s[idx_iter]).sum(), V[idx_iter, 0]))
    return V, pi_s

def plot_values(Vs_PI):
	plt.figure()
	plt.plot(Vs_PI)
	plt.xlabel("# iteration")
	plt.ylabel("state value")
	plt.legend([f"state {k}" for k in range(Vs_PI.shape[1])])
	plt.grid()
	plt.savefig("state_values_policy_iteration.jpg")
	

if __name__ == '__main__':
	GAMMA = 0.95
	env = gym.make("FrozenLake-v1").unwrapped
	
	P, R = convert_P(env)
	pi0 = np.arange(16) % env.action_space.n
	actual_val = compute_vpi(pi0, P, R, gamma=GAMMA)
	print("Policy Value: ", actual_val)
	
	Qpi = compute_qpi(np.arange(16), P, R, gamma=GAMMA)
	print("Policy Action Value: ", Qpi)
	
	Vs_PI, pis_PI = policy_iteration(env, gamma=GAMMA, num_iter=20)
	plot_values(Vs_PI)
	print(pis_PI)
	env.close()
 
