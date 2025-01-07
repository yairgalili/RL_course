"""
Bellman Equation:
Q*(s, a)=r(s,a) + γ sum_s' p(s'|s, a)V*(s')
V*(s)=max_a(Q*(s, a))

Q*(s, a)=r(s,a) + γ sum_s' p(s'|s, a)max_a(Q*(s', a))

where
r(s, a) = sum_s' (r(s, a, s') p(s'|s, a))
"""



"""
Clause a:
r' = c * r + b

Q*_M'(s, a)=cr(s,a) + b + γ sum_s' p(s'|s, a)max_a(Q*_M'(s', a))

Assume:
Q*_M'(s, a) = c' Q*_M(s, a) + b'

Therefore:
c' Q*_M(s, a) + b' = Q*_M'(s, a)=cr(s,a) + b + γ sum_s' {p(s'|s, a)max_a(c'Q*_M(s', a)+b')} = cr + b + γb' + γc' sum_s' p(s'|s, a)max_a(Q*_M(s', a))

From Bellman Equation:
Q*_M(s, a)=r(s,a) + γ sum_s' p(s'|s, a)max_a(Q*_M(s', a))

Therefore:
c' Q*_M(s, a) + b' = c'r + b' + γc' sum_s' p(s'|s, a)max_a(Q*_M(s', a))

From comparison of the two expressions:
c'=c
b' = b + γb' 
Hence:
b' = b/(1-γ)

Q*_M'(s, a) = c Q*_M(s, a) + b/(1-γ)

We proved that the optimal Q functions obey the aforementioned relation.

The optimal policy is given by:
	π* = argmax_a (Q*(s, a))

The optimal policies for M and M' are identical, since the relation between the Q functions is monotonically increasing (c>0).
"""


"""
Clause b:
r'(s, a, s') = r(s, a, s') + f(s) - γf(s')
see
Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping [1999]

We start with Bellman Equation:
	
Q*(s, a)=E_s' [r(s, a, s') +γ * max_a(Q*(s', a)) ]

E_s' [r(s, a, s') +γ * max_a(Q*(s', a)) ] = E_s' [r(s, a, s') + f(s) - γf(s') -(f(s) - γf(s'))+γ * max_a(Q*(s', a)) ] = E_s' [r'(s, a, s') -f(s) +γ * max_a(Q*(s', a) + f(s')) ]

Hence,
Q*(s, a) + f(s) = E_s' [r'(s, a, s') +γ * max_a(Q*(s', a) + f(s')) ]

Define,
Q'(s, a) = Q*(s, a) + f(s)

Q'(s, a) = E_s' [r'(s, a, s') + γ * max_a(Q'(s', a)) ]

Q*_M'(s, a) = Q'(s, a)
The function Q'(s, a) is the optimal Q function for M', since it obeys Bellman equation for the reward r'(s, a, s').

π*_M'(s) = argmax_a Q'(s, a) = argmax_a Q*(s, a) = π*_M(s)

The optimal policies for M and M' are identical.

"""