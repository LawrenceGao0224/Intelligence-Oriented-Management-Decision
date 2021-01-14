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
r = 0.0109
T = 22/365


def NewtownBLS(S,L,T,r,call):
    vol = 0.5
    while(abs(blsprice(S,L,T,r,vol)-call)>0.00001):
        vol = vol-(blsprice(S,L,T,r,vol)-call)/((blsprice(S,L,T,r,vol+0.00001)-blsprice(S,L,T,r,vol-0.00001))/0.00002)
    return vol
matrix = []
matrix.append(NewtownBLS(S,10900,T,r,173.00))
matrix.append(NewtownBLS(S,11000,T,r,115.00))
matrix.append(NewtownBLS(S,11100,T,r,71.00))
matrix.append(NewtownBLS(S,11200,T,r,40.50))
matrix.append(NewtownBLS(S,11300,T,r,22.00))
matrix.append(NewtownBLS(S,11400,T,r,11.50))
matrix.append(NewtownBLS(S,11500,T,r,6.00))
matrix.append(NewtownBLS(S,11600,T,r,3.20))
x = [10900,11000,11100,11200,11300,11400,11500,11600]
plt.plot(x,matrix)
plt.show()
