# Markov chain:
	
# Transition diagram:
"""
1->2, w.p. 1
2->3, w.p. 1/3
2->4, w.p. 2/3
3->1, w.p. 1
4->2, w.p. 1/2
4->3, w.p. 1/2
"""

"""
communicating classes:
If all the states in the Markov Chain belong to one closed communicating class, then the chain is called an irreducible Markov chain.
The states 1, 2, 3, 4 are in a single communicating class, therefore the chain is irreducible.
"""

"""
The period of a state i is the greatest common divisor of the set {n ∈ N : pn (i,i) > 0}. If every state has period 1 then the Markov chain (or its transition probability matrix) is called aperiodic.
In an irreducible Markov chain, all states have the same period d

d1=gcd(3, 4)=1
Therefore the Markov chain is aperiodic.
"""


"""
A probability distribution π ∈ M(X) is said to be invariant distribution for the Markov chain X if it satisfies the global balance equation π = πP.

Assume:
π = [x1, x2, x3, 1-x1-x2-x3]>=0
x1=x3
x2=x1+x4/2=x1+(1-2*x1-x2)/2=0.5(1-x2)
x3=x2/3+x4/2=x1=x2/3+(1-2*x1-x2)/2
Hence:
x2=1/3
x1=1/9+(1/3-x1)=4/9-x1
x1=2/9
π =[2, 3, 2, 2]/9
"""

"""
Let Ty = inf{n ≥ 1 : Xn = y}. Then E(Ty | X0 = y) = my is called the expected first return timefor the state y.
If a Markov chain has invariant distribution π then the expected return time to a persistent state i is 1/πi:
[4.5, 3, 4.5, 4.5]
"""

"""
New transition matrix Pso that:
 E[T1] = 3 and d1 = 3. Calculate p1,1(m)as afunction of m for that transition matrix given that 1 is the initial state.
 
1->2->3->1 w.p. 1
 p1,1(m)=1 iff 3|m o.w. 0
"""
import numpy as np

P_new = np.array([[0, 1, 0, 0],
[0, 0, 1, 0],
[1, 0, 0, 0],
[0, 0, 0, 1]])
print(P_new)
