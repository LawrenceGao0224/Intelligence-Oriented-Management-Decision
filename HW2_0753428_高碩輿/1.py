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
N = 100


M = 20000

matrix = []
S200 = []
S2000 = []
for i in range(M):
    Sa = MCsim(S,T,r,vol,N)
    matrix.append(Sa[99]) 
    if i <200:
        S200.append(matrix[i])
    if i <2000:
        S2000.append(matrix[i])
'''
x = range(200)
plt.bar(x, S200)
'''
'''
x = range(2000)
plt.bar(x, S2000)
'''

x = range(20000)
plt.bar(x, matrix)
