"""
Let
μ_b=μ_1, μ_a=μ_2
μ_mean = (μ_1+μ_2)/2
a)
sum_{It=2, μ2_hat>μ_mean}(P(μ2_hat>μ_mean, It=2))<=sum_{It=2}(P(μ2_hat>μ_mean))=sum_{It=2}(P(μ2_hat-μ_2>Δ/2))
<=sum_{It=2} exp{-T2(t)Δ^2/2}
Pay attention that T2(t) is an increasing sequence, i.e. all the elements T2(t) are different in the summation, since It=2 for all of them.
sum_{It=2} exp{-T2(t)Δ^2/2}<=sum_{n} exp{-nΔ^2/2}=1/(1-exp{-Δ^2/2})
"""



"""
b)
Pay attention that:
1) P(μ2_hat=μ_mean)=0, since it is a continuous distribution.

2) If μ2_hat<μ_mean:
If μ1_hat>μ_mean -> It=1.
If μ1_hat<μ_mean: play both arms one after another, It=1, I_{t+1}=2


sum_{It=2, μ2_hat<=μ_mean}(P(μ2_hat<=μ_mean, It=2))<=sum_{It=2,μ2_hat<=μ_mean}(P(μ2_hat<=μ_mean, μ1_hat<μ_mean))<=sum_{It=2,μ2_hat<=μ_mean}(P(μ1_hat<μ_mean))=sum_{It=2,μ2_hat<=μ_mean}(P(μ1_hat-μ_1<-Δ/2)) <= sum_{It=2,μ2_hat<=μ_mean}exp{-T1(t)Δ^2/2}


Pay attention that T1(t) is an increasing sequence, i.e. all the elements T1(t) are different in the summation, since the cases when It=2, μ1_hat<μ_mean, μ2_hat<μ_mean occur only if It=2, I_{t-1}=1, therefore the value of T1(t) was altered.

<=sum_{n} exp{-nΔ^2/2}=1/(1-exp{-Δ^2/2})

"""

"""
c)
sum_{t>2} p(It=2)<=2/(1-exp{-Δ^2/2})
p(I2=2)=1, p(I1=2)=0

 E[ sum_t μ_It ] = sum_t {p(It=2)μ2+p(It=1)μ1}
 
 R = Tμ1-E[sum_t(μ_It)] = Tμ1-sum_t {p(It=2)μ2+(1-p(It=2))μ1} = sum_t {p(It=2)*(μ1-μ2)} = Δ(1+sum_{t>2} p(It=2))<=Δ(1+2/(1-exp{-Δ^2/2}))
 
 """