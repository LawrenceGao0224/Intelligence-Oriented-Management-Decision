import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

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
    tc =  np.random.randint(0,512)
    w = np.random.random(4,8)
    m = np.random.random(0,1)
    phi = np.random.random(0,math.pi)
    tm = np.power(tc - t, m)
    return np.exp(a + b*tm + c*tm*np.cos(w*np.log(tc-t)-phi))

n = 1000
A = np.zeros((n,5))
b = np.zeros((n,1))
for i in range(n):
    t = np.random.random()*100
    b[i] = F1(t)
    A[i,0] = t **4
    A[i,1] = t **3
    A[i,2] = t **2
    A[i,3] = t 
    A[i,4] = 1

X = np.linalg.lstsq(A,b)[0]
print(X)
S0 = np.arange(-5.11,5.13,0.01)
T = np.random.random((n,1))*100
b2 = F2(T,0.6,1.2,100,0.4)

a = []
for i in frange(-5.11,5.12,0.01):
    b2 = F2(T,0.6,1.2,100,i)
    a.append(Energy(b2,T,0.6,1.2,100,i))
plt.plot(S0,a)
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('A')
ax.set_ylabel('C')
ax.set_zlabel('Energy')
X = np.arange(-5.11, 5.13, 0.01)
Y = np.arange(-511, 513, 1)
X, Y = np.meshgrid(X, Y)
c = [[0]*1024 for i in range(1024) ]
k=0
for i in frange(-5.11,5.12,0.01):
    l=0
    for j in range(-511,513):
        b2 = F2(T,i,1.2,j,0.4)
        c[k][l] = Energy(b2,T,i,1.2,j,0.4)
        l+=1
    k+=1
surf = ax.plot_surface(X, Y, c, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


'''GA
pop = np.random.randint(0,2,(10000,40)) #隨機產生10000人,40bit


fit = np.zeros((10000,1))

for generation in range(10):#跑10代
    for i in range(10000): #評估活的好壞
        gene = pop[i,:]
        A = (np.sum(2**np.arange(10)*gene[0:10])-511)/100
        B = (np.sum(2**np.arange(10)*gene[10:20])-511)/100
        C = (np.sum(2**np.arange(10)*gene[20:30])-511)
        D = (np.sum(2**np.arange(10)*gene[30:40])-511)/100
        fit[i] = Energy(b2,T,A,B,C,D)
    sortf = np.argsort(fit[:,0])
    pop = pop[sortf,:]#基因由小到大排序
    for i in range(100,10000):#覆蓋100~10000
        fid = np.random.randint(0,100) #從0~99取一個當爸爸
        mid = np.random.randint(0,100) #從0~99取一個當媽媽
        while mid == fid:
            mid = np.random.randint(0,100)
        mask = np.random.randint(0,2,(1,40))
        son = pop[mid,:] #copy 媽媽的基因
        father = pop[fid,:]
        son[mask[0,:]==1]== father[mask[0,:]==1] #copy 爸爸的基因
        pop[i,:]= son
    for i in range(1000):#突變多少人
        m = np.random.randint(0,10000)
        n = np.random.randint(0,40)    #地m個人地n個基因
        pop[m,n] = 1-pop[m,n]
        
for i in range(10000): #評估活的好壞
    gene = pop[i,:]
    A = (np.sum(2**np.arange(10)*gene[0:10])-511)/100
    B = (np.sum(2**np.arange(10)*gene[10:20])-511)/100
    C = (np.sum(2**np.arange(10)*gene[20:30])-511)
    D = (np.sum(2**np.arange(10)*gene[30:40])-511)/100
    fit[i] = Energy(b2,T,A,B,C,D)
sortf = np.argsort(fit[:,0])
pop = pop[sortf,:]
        
gene = pop[0,:]
A = (np.sum(2**np.arange(10)*gene[0:10])-511)/100
B = (np.sum(2**np.arange(10)*gene[10:20])-511)/100
C = (np.sum(2**np.arange(10)*gene[20:30])-511)
D = (np.sum(2**np.arange(10)*gene[30:40])-511)/100
print(A,B,C,D)


data = np.load('data.npy')
plt.plot(data)
'''