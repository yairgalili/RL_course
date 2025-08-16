"""
An Undiscounted Markov Reward Process (MRP) is a special case of a Markov Decision Process (MDP) where:

- There are no actions (or a fixed policy is followed).
- The process is stochastic, but transitions depend only on the current state (Markov property).
- The reward is accumulated over time, and no discounting is applied.

Formal Definition

An MRP is defined by a tuple (S, P, R) where:

- S: Set of states.
- P(s' | s): Transition probability from state s to s'.'
- R(s): Expected immediate reward when in state s.

In the undiscounted case:

- The discount factor γ = 1.
- The total return (value) from state s is:

  V(s) = E[Σₜ R(Sₜ) | S₀ = s]

  where the sum is over infinite or finite horizon, depending on the setting.

a) Empirical model:
	The estimated transition matrix for states S_A, S_B and terminated.
	E.g.
	6 cases of S_A->S_B, 4 cases of S_A->terminated
	P = [0, 7/12, 5/12;
	7/12, 0, 7/12;
	0, 0, 1]
	E[R(S_A)] = (5*2)/12=5/6
	E[R(S_B)] = 7/12
	
	
b)
To estimate V(S_A) and V(S_B) using first-visit Monte Carlo (MC) evaluation, we:

1. Identify first visits to each state in every episode.
2. For each first visit, record the return G (sum of rewards until termination).
3. Average these returns across all first visits for each state.

V(S_A) = 14/9
V(S_B) = 15/11

---



"""