"""
a)
P(0<a<1)=int(λe^(-λx), 0, 1) = 1-e^(-λ) = 1-e^(-θ^T*Φ(s))
"""


"""
b)
grad_θ log π(a|s, θ) = grad_λ(log λ- aλ)Φ(s) = (1/λ- a)Φ(s) 


λ=θ^T*Φ(s)

"""


"""
c)
General REINFORCE Update:
For a parameterized distribution π(a|θ), the REINFORCE update is:
J(θ) = E_π[R]
∇_θ J(θ) = E_π[∇_θ log π(a|θ) · R]
where R is the return reward.

In our case:
θ_new = θ_old + α E_π[(1/λ- a)Φ(s) · R]

The expectation is computed by sampling according to the current policy and averaging the results.

"""