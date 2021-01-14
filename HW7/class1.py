# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 19:12:28 2018

@author: rsps971130
"""

import numpy as np 
import math
import random
import matplotlib.pyplot as plt
#def f(x1,x2):
#    if (x1*1 + x2*2 > 0):
#        h1 = 1
#    else:
#        h1 = -1
#    if (x1*3 + x2*1 > 0):
#        h2 = 1
#    else:
#        h2 = -1
#    if (x1*1 + x2*1 > 0):
#        h3 = 1
#    else:
#        h3 = -1
#    if (h1*2 + h2*1 - h3*3) > 0:
#        return 1
#    else:
#        return -1
#    
#x = np.zeros((21,21))
#for i in range(-10,11):
#    for j in range(-10,11):
#        x[i+10][j+10] = f(i,j)
#   

npzfile = np.load('CBCL.npz')  
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']
def drange(start, stop, step):
     r = start
     while r < stop:
         yield r
         r += step
def bpnntrain(pf,nf,hn,lr,iteration):
    pn,fn = pf.shape
    nn = nf.shape[0]
    feature = np.append(pf,nf,axis=0)
    target = np.append(np.ones((pn,1)),np.zeros((nn,1)))
    WI = np.random.normal(0,1,(fn+1,hn))
    WO = np.random.normal(0,1,(hn+1,1))
    for t in range(iteration):
        s = random.sample(range(pn+nn),pn+nn)
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1)
            ho = ins.dot(WI)
            ho = 1/(1+np.exp(-ho))
            hs = np.append(ho,1)
            out = hs.dot(WO)
            out = 1/(1+np.exp(-out))
            dk = out*(1-out)*(target[s[i]]-out)
            dh = ho*(1-ho)*WO[:hn,0]*dk
            WO[:,0] += lr*dk*hs
            for j in range(hn):
                WI[:,j] += lr*dh[j]*ins
    model = dict()
    model['WI'] = WI
    model['WO'] = WO
    return model


def bpnntest(feature,model):
    sn = feature.shape[0]
    WI = model['WI']
    WO = model['WO']
#    hn = WI.shape[1]
    out = np.zeros((sn,1))
    for i in range(sn):
        ins = np.append(feature[i,:],1)
        ho = ins.dot(WI)
        ho = 1/(1+np.exp(-ho))
        hs = np.append(ho,1)
        out[i] = hs.dot(WO)
        out[i] = 1/(1+np.exp(-out[i]))
    return out
x = []
y = []
'''
for i in range(1,20,1):#調整hidden node
    network = bpnntrain(trainface/255,trainnonface/255,i,0.01,10)
    pscore = bpnntest(testface/255,network)
    nscore = bpnntest(testnonface/255,network)
    x.append(np.sum(pscore>0.5)/len(pscore))
    y.append(np.sum(nscore>0.5)/len(nscore))
plt.plot(x,y, 'ro')
plt.xlabel('pscore')
plt.ylabel('nscore')
plt.show()
'''

network = bpnntrain(trainface/255,trainnonface/255,20,0.016,20)
pscore = bpnntest(testface/255,network)
nscore = bpnntest(testnonface/255,network)
print(np.sum(pscore>0.5)/len(pscore))
print(np.sum(nscore>0.5)/len(nscore))

'''
for i in range(10,20,1):#調整interation
    network = bpnntrain(trainface/255,trainnonface/255,20,0.01,i)
    pscore = bpnntest(testface/255,network)
    nscore = bpnntest(testnonface/255,network)
    x.append(np.sum(pscore>0.5)/len(pscore))
    y.append(np.sum(nscore>0.5)/len(nscore))
plt.plot(x,y, 'ro')
plt.xlabel('pscore')
plt.ylabel('nscore')
plt.show()
'''
'''
for i in drange(0.01,0.02,0.001):#learning rate
    network = bpnntrain(trainface/255,trainnonface/255,20,i,10)
    pscore = bpnntest(testface/255,network)
    nscore = bpnntest(testnonface/255,network)
    x.append(np.sum(pscore>0.5)/len(pscore))
    y.append(np.sum(nscore>0.5)/len(nscore))
plt.plot(x,y, 'ro')
plt.xlabel('pscore')
plt.ylabel('nscore')
plt.show()
'''
'''
for i in drange(0.1,0.9,0.1):#threshold
    network = bpnntrain(trainface/255,trainnonface/255,20,0.01,10)
    pscore = bpnntest(testface/255,network)
    nscore = bpnntest(testnonface/255,network)
    x.append(np.sum(pscore>i)/len(pscore))
    y.append(np.sum(nscore>i)/len(nscore))
plt.plot(x,y, 'ro')
plt.xlabel('pscore')
plt.ylabel('nscore')
plt.show()
'''
    
    
    
    
    