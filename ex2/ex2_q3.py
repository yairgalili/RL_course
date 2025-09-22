"""
take a random digit:
	put in an unoccupied place
	
Maximize:
	sum(di*10^i)
	
S={(ai) | ai is digit or empty place, d is chosen}
A = decide on digit place
P(s_(t+1)|st, a) = after deciding on digit place w.p. 0.1 we get each new (ai) which a digit that was chosen.
R = sum(di*10^i) iff there is no empty places
s0 = (x,x,x,...,x) - empty places
"""


"""
The decision of the optimal policy depends only on the number of empty

slots and the current random digit di.
We can prove the claim by defining an isomorphism between MDPs, φ(s)=u, Φ(a) which obey:
1) P(s_(t+1)|st, a) = P(u_(t+1)|ut, a)
2) R(φ(s1), a)<=R(φ(s2), a) iff R(s1 a)<=R(s2, a)

The optimal policy depends solely on transition propabilities and rewards order and not on the value of the reward.
If a policy maximizes an expected reward, then it maximizes every monotonic non decreasing function of the reward.

We can map xx12 to xx to 1x2x etc.
Therefore the decision of the optimal policy depends only on the number of empty
slots and the current random digit di.
"""


"""
Optimal policy for N = 2, i.e. the number has three digits.

xxx, d ->
	 (xxa) iff a<=3
	 (xax) iff 4<=a<=5
	 (axx) iff a>=6
xx, d -> (dx) if d>=5 else (xd)
x, d -> (d), single place for digit

Vk^*(s) = max_π E[R|sk=s]
π*_t(s) = argmax_a (rt(s, a) + sum(pt(s'|s, a')V_(t+1)(s'')))

Let:
s = (xx, a), we have two possible actions (ax, d) or (xa, d), the expected cost for each action:
sum_d(E[R|(ax, d)])/10 = sum_d(10*a+d)/10 = 5(20a+9)/10
sum_d(E[R|(xa, d)])/10 = sum_d(10*d+a)/10 = 5(90+2a)/10

20a+9>90+2a iff a>4.5
Therefore iff a>=5: decide (ax), o.w. decide (xa).
V*((xx, a)) = 0.5*max(90+2a, 20a+9)
V*((xbx, a)) = max(sum_d(E[R|(abx, d)])/10, sum_d(E[R|(xba, d)])/10) = 10b + 0.5*max(200a+9, 2a + 900)
V*((xxb, a)) = max(sum_d(E[R|(axb, d)])/10, sum_d(E[R|(xab, d)])/10) = b + 0.5*max(200a+90, 20a + 900)

Let:
s = (xxx, a), we have three possible actions (axx, d) or (xax, d) or (xxa, d) the expected cost for each action:

1) sum_d(E[R|(axx, d)])/10 = 100*a + sum_d(max(5(90+2d)/10, 5(20d+9)/10))/10 = 100a + 60.75

2) sum_d(V*((xax, d)))/10 = 10*a + sum_d(0.5*max(200d+9, 2d + 900))/10 = 10a + 0.05*(sum(200d+9, d=5, 9) + sum(2d+900, d=0, 4)) = 10a + 0.05*(1009+1809+900+908)*5/2 = 10a + 578.25

3) sum_d(V*((xxa, d)))/10 = a + sum_d(0.5*max(200d+90, 20d + 900))/10 = a + 0.05*(sum(200d+90, d=5, 9) + sum(20d+900, d=0, 4)) = a + 0.05*(1090+1890+900+980)*5/2 = a + 607.5

100a + 60.75 > 10a + 578.25 iff a>5.75
100a + 60.75 > a + 607.5 iff a>5.52
10a+578.25 > a+607.5 iff a>3.25

decide:
	 (xxa) iff a<=3
	 (xax) iff 4<=a<=5
	 (axx) iff a>=6
"""
import numpy as np

def get_value(x):
	return sum(np.flip(x) * (10 ** np.arange(len(x))))

def V(x, new_element):
	num_missing = sum(x == -1)
	if num_missing == 1:
		# (x, d) case
		x_copy = x.copy()
		x_copy[x_copy==-1] = new_element
		return get_value(x_copy)
	
	# max(sum_d(V*((xax, d)))/10, ...)
	ind_missing = np.argwhere(x == -1)
	res = []
	for ind in ind_missing:
		x_copy = x.copy()
		x_copy[ind] = new_element
		res.append([np.sum([V(x_copy, d) for d in range(10)]) / 10])
		
	ind_policy = np.argmax(res)
	print(f"The optimal policy for {x} is to put {new_element} in the index: {ind_missing[ind_policy]}")
	return np.max(res)
		
		
		
		
if __name__ == '__main__':
	# one missing
	print(V(np.array([2, -1]), 5))
	print(V(np.array([-1, 2, 4]), 7))
	
	# two missings: 45+a
	print(V(np.array([-1, -1]), 1))
	
	# V*((bxx, a)) = 100b + 0.5*max(90+2a, 20a+9)
	print(V(np.array([5, -1, -1]), 7))
	
	# V*((xbx, a)) = 10b + 0.5*max(200a+9, 2a + 900)
	print(V(np.array([ -1, 5, -1]), 7))
	
	# V*((xxb, a)) = b + 0.5*max(200a+90, 20a + 900)
	print(V(np.array([ -1, -1, 3]), 7))
	
	
	# V*((xxx, d))
	print(V(np.array([ -1, -1, -1]), 7)) 
	print(V(np.array([ -1, -1, -1]), 2))
	print(V(np.array([ -1, -1, -1]), 4))
	
	print(V(np.array([ -1, -1, -1, -1]), 4))
	
	

	
	
	