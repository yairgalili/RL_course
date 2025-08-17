"""

The temporal difference (TD) error, denoted as Δₜ, measures the difference between the current estimate of the value of a state and a revised estimate based on observed reward and next state's value.
Δₜ = rₜ + γ * V(Sₜ₊₁) − V(Sₜ)

Where:
- rₜ: reward received after taking action at time t  
- γ: discount factor (0 ≤ γ ≤ 1)  
- V(Sₜ): estimated value of current state  
- V(Sₜ₊₁): estimated value of next state

---
Intuition:
- If Δₜ > 0, it means the outcome was better than expected → increase V(Sₜ)
- If Δₜ < 0, it means the outcome was worse than expected → decrease V(Sₜ)
- Used in TD learning methods like TD(n), SARSA, and Q-learning

Bellman Equation:
	V^π(s)=E[rt+γ*V^π(S_{t+1})|St=s]
	
E[Δt] = 0 under the true policy.
It means that there is no learning signal where the value estimates are perfect.

b)
E[Δₜ | Sₜ = s, Aₜ = a] = E[rₜ + γ * V^π(Sₜ₊₁) − V^π(s) | Sₜ = s, Aₜ = a]

Q^π(s, a) = E[rₜ + γ * V^π(Sₜ₊₁) | Sₜ = s, Aₜ = a]
V^π(s) = E_{a'∼π}[Q^π(s, a')]

Therefore:

E[Δₜ | Sₜ = s, Aₜ = a] = Q^π(s, a) − V^π(s)

- The expected TD error under off-policy action a is the advantage of taking action a compared to following policy π.
- If Q^π(s, a) > V^π(s)→ TD error is positive: action a is better than π would expect.
- If Q^π(s, a) < V^π(s) → TD error is negative: action a is worse.
- This is the core of advantage estimation, often used in policy gradient methods and actor-critic algorithms.

In essence, TD error reflects the performance gap between the taken action and what the policy π would have done — a key quantity in off-policy learning.

"""