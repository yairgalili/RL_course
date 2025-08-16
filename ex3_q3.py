"""
Consider a DDP (Deterministic Decision Process) that is modeled by a directed graph, where
the states are the vertices, and each action is associate with an edge with unknown reward
function r and Rmax = 1.
s_new = f(s_old, a) is the new state after taking the action a.

For DDP:
	
V(s) = min_a {r(s, a) + γ * V(f(s, a))}
π*(s) = argmin_a {r(s, a) + γ * V(f(s, a))}

Algorithm for discovering DDP:
	for iter t:
	  1) initialize:
	  r'(s, a) = 1 if (s, a) is unknown else 0
	  f(s, a) = s if (s, a) is unknown
	  2) find optimal policy π_t
	  3) run it until it discovers a new state-action
	  

a) Every iteration terminates:
	The agent gets a reward iff it executes an unknown (s, a).
If we know n states and they have an unknown (s, a) pair, we can execute it.
If all their states, actions pairs are known then we discovered all the DDP, since it is strongly connected.
The optimal policy is doing the minimal number of transitions for reaching unknown (s, a).
Assume by contradiction, that the agent does not know the path to unrecovered (s, a).
Therefore, from s1 you can reach S states, that their states (s, a) are known, and their (s, a) are known and etcetera, so you know to reach all the states since the network is strongly connected, in contradiction.
b) Maximal time complexity is |S|-1, since we need to go from s1 to s2, where the path is known and we do not have cycles.
c) Number of iterations: |S|*|A|
d) T<= |S|*|A|*(|S|-1)

"""