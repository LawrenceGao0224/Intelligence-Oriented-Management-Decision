import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def blsprice(S,L,T,r,vol):
    d1 = (math.log(S/L)+(r+0.5*vol*vol)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    call =  S *norm.cdf(d1)-L*math.exp(-r*T)*norm.cdf(d2)
    return call

S = 10978.85
L = 11200
r = 0.0109
T = 22/365
call = 40.5

def BisectionBLS(S,L,T,r,call):
    left = 0.00000001
    right = 1
    matrix =  []
    count = 0
    while(right-left>0.0000001):
        middle = (left+right)/2
        count += 1
        if count <= 20:
            matrix.append(middle)
        else:
            break
        if((blsprice(S,L,T,r,middle)-call)*(blsprice(S,L,T,r,left)-call)<0):
            right = middle
        else:
            left = middle
    return matrix

def NewtownBLS(S,L,T,r,call):
    vol = 0.5
    count = 0
    matrix2 = []
    
    while(abs(blsprice(S,L,T,r,vol)-call)>0.00000001): 
        vol = vol-(blsprice(S,L,T,r,vol)-call)/((blsprice(S,L,T,r,vol+0.00000001)-blsprice(S,L,T,r,vol-0.00000001))/0.00000002)
        count += 1
        if count <= 20: 
            matrix2.append(vol)
    return matrix2
P1 = BisectionBLS(S,L,T,r,call)
P2 = NewtownBLS(S,L,T,r,call)
plt.plot(P1)
plt.plot(P2)
plt.show()



