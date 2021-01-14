import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
import random

def F1(t):
    return 0.063*(t**3)-5.284*(t**2)+4.887*t+412+np.random.normal(0,1)
def F2(t,A,B,C,D):
    return A*(t**B)+C*np.cos(D*t)+np.random.normal(0,1,t.shape)
def Energy(b2,T,A,B,C,D):
    return np.sum(abs(b2-F2(T,A,B,C,D)))
def frange(start, stop, step):
     x = start
     while x < stop:
         yield x
         x += step
def LPPL(a,b,c):
    return (a + b*tm + c*tm*np.cos(w*np.log(tc-t)-phi))
def LPPL2(a,b,c,tc,w,m,phi,t):
    tm = np.power(tc - t, m)
    return (a + b*tm + c*tm*np.cos(w*np.log(tc-t)-phi))
def Distance(LPPL,data):
    return np.sum(abs(LPPL-data))
def Linear(tc,w,m,phi):
    global X
    n = 500
    A = np.zeros((n,3))
    B = np.zeros((n,1))
    for t in range(n):
        tm = np.power(tc - t, m)
        B[t] = data[t]
        A[t,0] = 1
        A[t,1] = tm
        A[t,2] = tm*np.cos(w*np.log(tc-t)-phi)
    X = np.linalg.lstsq(A,B)[0]
    return X
'''跑first time linear 求出 A,B,C'''
data = np.load('data.npy')
n = 500
A = np.zeros((n,3))
B = np.zeros((n,1))
for t in range(n):
    tc =  random.randint(500,515)
    w = random.uniform(6, 13)
    m = random.uniform(0.1,0.9)
    phi = random.uniform(0,2*(math.pi))
    tm = np.power(tc - t, m)
    B[t] = data[t]
    A[t,0] = 1
    A[t,1] = tm
    A[t,2] = tm*np.cos(w*np.log(tc-t)-phi)
X = np.linalg.lstsq(A,B)[0]
print(X)



'''GA'''
T = np.random.random((n,1))*100
b2 = data


def NonLinear(A,B,C):
    global tc
    global w
    global m
    global phi
    pop = np.random.randint(0,2,(10000,34)) #隨機產生10000人,40bit
    fit = np.zeros((10000,1))
    L = LPPL(A,B,C)
    for generation in range(20):#跑10代
        for i in range(10000): #評估活的好壞
            gene = pop[i,:]
            tc =  np.sum(2**np.arange(4)*gene[0:4])+500
            w = (np.sum(2**np.arange(10)*gene[4:14])-6)/1023*7
            m = (np.sum(2**np.arange(10)*gene[14:24])-0.1)/1023*0.8
            phi = (np.sum(2**np.arange(10)*gene[24:34])/1023)*2*(math.pi)
            fit[i] = Distance(L,data)
        sortf = np.argsort(fit[:,0])
        pop = pop[sortf,:]#基因由小到大排序
        for i in range(100,10000):#覆蓋100~10000
            fid = np.random.randint(0,100) #從0~99取一個當爸爸
            mid = np.random.randint(0,100) #從0~99取一個當媽媽
            while mid == fid:
                mid = np.random.randint(0,100)
            mask = np.random.randint(0,2,(1,34))
            son = pop[mid,:] #copy 媽媽的基因
            father = pop[fid,:]
            son[mask[0,:]==1]== father[mask[0,:]==1] #copy 爸爸的基因
            pop[i,:]= son
        for i in range(1000):#突變多少人
            m = np.random.randint(0,10000)
            n = np.random.randint(0,34)    #地m個人地n個基因
            pop[m,n] = 1-pop[m,n]
            
    for i in range(10000): #評估活的好壞
        gene = pop[i,:]
        tc =  np.sum(2**np.arange(4)*gene[0:4])+500
        w = (np.sum(2**np.arange(10)*gene[4:14])-6)/1023*7
        m = (np.sum(2**np.arange(10)*gene[14:24])-0.1)/1023*0.8
        phi = (np.sum(2**np.arange(10)*gene[24:34])/1023)*2*(math.pi)
        fit[i] = Distance(L,data)
    sortf = np.argsort(fit[:,0])
    pop = pop[sortf,:]
            
    gene = pop[0,:]
    tc =  np.sum(2**np.arange(4)*gene[0:4])+500
    w = (np.sum(2**np.arange(10)*gene[4:14])-6)/1023*7
    m = (np.sum(2**np.arange(10)*gene[14:24])-0.1)/1023*0.8
    phi = (np.sum(2**np.arange(10)*gene[24:34])/1023)*2*(math.pi)
    return tc,w,m,phi
'''執行'''
for i in range(20):
    NonLinear(X[0],X[1],X[2])
    Linear(tc,w,m,phi)
    print(NonLinear(X[0],X[1],X[2]))
    print(Linear(tc,w,m,phi))
    
a =[]
for t in range(0,500):
    a.append(LPPL2(X[0],X[1],X[2],tc,w,m,phi,t))
plt.plot(data)
plt.plot(a)
plt.show()