"""
S = {i mod 2k}
A = {CW, CCW}
P_ij(CW) = δ(j, i+1mod2k)
P_ij(CCW) = δ(j, i-1mod2k)
Rt(s,a) = δ(s, 0)
s0 = k
"""


"""
optimal policy is π*
k->k-1->...->0->1->0->...
"""

"""
After one iteration of the Value Iteration algorithm (where all states start with a valueof 0):
V_n+1(s) = max_a{r(s,a)+γΣ_s'(p(s'|s,a)V_n(s'))}
V1(s)=δ(s, 0)
Only the zero state changes its value.
"""


"""
After two iterations of the Value Iteration algorithm:
For s!=0:
V2(s) = γ*max_a(p(0|s, a)) = γ(δ(s, 1) + δ(s, 2k - 1))
V2(0) = 1 + γ*max_a(p(0|0, a)) = 1
The states 1, 2k-1 change their value.
"""

"""
Bellman Optimality Equation:
V*(s) = max_a {r(s, a) + γΣ_s'(p(s'|s, a)V*(s'))}
For k = 2 (4 states), V*(s) obey:
V*(0) = 1+γ*max(V*(1), V*(3)) = x + 1=1+γy
V*(1) = γ*max(V*(0), V*(2)) = y=γ(1+x)
V*(2) = γ*max(V*(1), V*(3)) = x=γy
V*(3) = γ*max(V*(2), V*(0)) = y=γ(1+x)
x=γy = γ^2(x+1)
y=γ(1+x)
x=γ^2/(1-γ^2)
y=γ(1+γ^2/(1-γ^2))=γ/(1-γ^2)
V*=[1, γ, γ^2, γ]/(1-γ^2)
"""