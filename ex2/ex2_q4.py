"""
We will show that T is a γ-contracting with respect to the max-norm.
We need to show:	
for all v1, v2: max(abs(Tv1-Tv2))<=γ*max(abs(v1-v2))

max(abs(Tv1-Tv2)) = γ/|A| * max(abs(sum_(a, s') (p(s'|s, a)(v1(s')-v2(s'))))) <= γ/|A| * max(sum_(a) |sum_ s' (p(s'|s, a)(v1(s')-v2(s')))|)

By claim 1, p_s'=p(s'|s, a):
|sum_ s' (p(s'|s, a)(v1(s')-v2(s')))|<= 
max |v1(s')-v2(s')|

γ/|A| * max(sum_(a) |sum_ s' (p(s'|s, a)(v1(s')-v2(s')))|) <= γ/|A| * max_s(sum_(a) max_s' |v1(s')-v2(s')|) = γ * max_(s, s') |v1(s')-v2(s')| = γ * max(|v1-v2|)
Q.E.D.
"""


"""
Claim 1:
if: p_s'>=0, sum(p_s')=1:
|sum_(s')(p_s' * v_s')| <=sum_(s')p_s' * |v_s'| <= max(|v_s'|)
"""