"""
Karp’s minimum mean-weight cycle algorithm - Proof in Problem 9-2. 

https://courses.csail.mit.edu/6.046/fall01/handouts/ps9sol.pdf

correction:
c)  theweight of the path from v to u along the cycle is x.
"""

import numpy as np

# d_k(v) is the minimal cost of paths of length k from root to v

def Bellman_Ford(A: np.array, d0: np.array, k: int):
	d_all = np.zeros((k+1, A.shape[0]))
	d_all[0] = d0
	d = np.reshape(d0, (-1, 1))
	for ind in range(1, k+1):
		d = np.min(d + A, axis=0)
		d_all[ind] = d
		d = np.reshape(d, (-1,1))
		print(f'd_{ind}:', d)
	print('d_all:', d_all)
	return d_all

def min_mean_cycle(d_all: np.array):
	diff_d = d_all[-1] - d_all[:-1]
	n = d_all.shape[1]
	mu = diff_d / np.reshape(n - np.arange(n), (-1, 1))
	
	mu = np.nan_to_num(mu, nan=np.inf)
	mu_opt = np.min(np.max(mu, axis=0))
	return mu_opt

if __name__ == '__main__':
	A = np.array([[np.inf, -1, np.inf, np.inf, np.inf],
	[np.inf, np.inf, 3, 2, 2],
	[4, np.inf, np.inf, np.inf, np.inf],
	[np.inf, 1, 5, np.inf, np.inf],
	[np.inf, np.inf, np.inf, -3, np.inf]])
	
	d0 = np.array([0, np.inf, np.inf, np.inf, np.inf])
	
	k = 5
	d_all = Bellman_Ford(A, d0, k)
	
	print('mu_opt:', min_mean_cycle(d_all))
	# mu_opt=0, since DBED cycle has cost 0.
	# optimal average cost = mu_opt = 0
	
