import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
def blsprice(S,L,T,r,vol):
    d1 = (math.log(S/L)+(r+0.5*vol*vol)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    call =  S *norm.cdf(d1)-L*math.exp(-r*T)*norm.cdf(d2)
    return call

def MCsim(S,T,r,vol,N):
    dt = T/N
    St = np.zeros((N+1))
    St[0] = S
    for i in range(N):
        St[i+1] = St[i]*math.exp((r-0.5*vol*vol)*dt+np.random.normal()*vol*math.sqrt(dt))
    return St
    
S = 50
L =40
T = 2
r = 0.08
vol = 0.2
S0 = np.arange(1,100,0.1)


a = 100
b = 1000
c = 10000



call2 =0
call3 =0
call4 =0
M = 100000
for i in range(M):
    Sa = MCsim(S,T,r,vol,a)
    if i <1000:
        if(Sa[-1]-L>0):
            call2 += (Sa[-1]-L)
         
    if i <10000:
        if(Sa[-1]-L>0):
            call3 += (Sa[-1]-L)
    if i <100000:
        if(Sa[-1]-L>0):
            call4 += (Sa[-1]-L)
print("分100期,模擬1000次: ",abs(call2/1000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))
print("分100期,模擬10000次: ",abs(call3/10000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))
print("分100期,模擬100000次: ",abs(call4/100000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))



call2 =0
call3 =0
call4 =0
M = 100000
for i in range(M):
    Sa = MCsim(S,T,r,vol,b)
    if i <1000:
        if(Sa[-1]-L>0):
            call2 += (Sa[-1]-L)
         
    if i <10000:
        if(Sa[-1]-L>0):
            call3 += (Sa[-1]-L)
    if i <100000:
        if(Sa[-1]-L>0):
            call4 += (Sa[-1]-L)
print("分1000期,模擬1000次: ",abs(call2/1000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))
print("分1000期,模擬10000次: ",abs(call3/10000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))
print("分1000期,模擬100000次: ",abs(call4/100000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))


call2 =0
call3 =0
call4 =0
M = 100000
for i in range(M):
    Sa = MCsim(S,T,r,vol,c)
    if i <1000:
        if(Sa[-1]-L>0):
            call2 += (Sa[-1]-L)
         
    if i <10000:
        if(Sa[-1]-L>0):
            call3 += (Sa[-1]-L)
    if i <100000:
        if(Sa[-1]-L>0):
            call4 += (Sa[-1]-L)
print("分10000期,模擬1000次: ",abs(call2/1000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))
print("分10000期,模擬10000次: ",abs(call3/10000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))
print("分10000期,模擬100000次: ",abs(call4/100000*math.exp(-r*T)-blsprice(S,L,T,r,vol)))