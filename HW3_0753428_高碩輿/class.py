# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:07:49 2018

@author: rsps971130
"""

import math
import numpy as np

#discrete
def entropy(p1,n1):
    if (p1 == 0 and n1 == 0):
        return 1
    elif(p1 == 0 or n1 == 0):
        return 0
    pp = p1 / (p1 + n1)
    pn = n1 / (p1 + n1)
    return -pp*math.log2(pp) - pn*math.log2(pn)   

def IG(p1,n1,p2,n2):
    num = p1 + n1 + p2 + n2
    num1 = p1 + n1
    num2 = p2 + n2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)

data = np.loadtxt('PlayTennis.txt',dtype=int)
feature = data[:,0:4]
target = data[:,4]-1

node = dict()
node['data'] = np.arange(len(target),dtype=int)
Tree = []
Tree.append(node)
t = 0
while(t<len(Tree)):
    idx = Tree[t]['data']
    if(sum(target[idx])==0):
        Tree[t]['leaf']=1
        Tree[t]['decision']=0
    elif(sum(target[idx])==len(idx)):
        Tree[t]['leaf']=1
        Tree[t]['decision']=1
    else:
        bestIG = 0
        for i in range(feature.shape[1]):
            pool = list(set(feature[idx,i]))
            pool.sort()
            for j in range(len(pool)-1):
                thres = (pool[j]+pool[j+1])/2
                G1 = []
                G2 = []
                for k in idx:
                    if(feature[k][i]<=thres):
                        G1.append(k)
                    else:
                        G2.append(k)
                thisIG = IG(sum(target[G1]==1),sum(target[G1]==0),sum(target[G2]==1),sum(target[G2]==0))
                if(thisIG>bestIG):
                    bestIG = thisIG
                    bestG1 = G1
                    bestG2 = G2
                    bestthres = thres
                    bestf = i
        if(bestIG>0):
            Tree[t]['leaf'] = 0
            Tree[t]['selectf'] = bestf
            Tree[t]['threshold'] = bestthres
            Tree[t]['child'] = [len(Tree),len(Tree)+1]
            node = dict()
            node['data']=bestG1
            Tree.append(node)
            node = dict()
            node['data']=bestG2
            Tree.append(node)
        else:
            Tree[t]['leaf'] = 1
            if(sum(target[idx]==1)>sum(target[idx]==0)):
                Tree[t]['decision']=1
            else:
                Tree[t]['decision']=0
            
    t = t+1    
test_feature = feature[13,:]
now = 0
while(Tree[now]['leaf']==0):
    bestf = Tree[now]['selectf']
    thres = Tree[now]['threshold']
    if(test_feature[bestf]<=thres):
        now = Tree[now]['child'][0]
    else:
        now = Tree[now]['child'][1]    
print(Tree[now]['decision'])        
                
                
                
                
                
                
                
                
            







