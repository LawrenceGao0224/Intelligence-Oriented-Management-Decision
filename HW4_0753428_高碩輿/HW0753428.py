from sklearn import datasets
import numpy as np
import random
import math
iris = datasets.load_iris()
X = iris.data

X1 =(X-np.tile(X.mean(0),(X.shape[0],1)))/np.tile(X.std(0),(X.shape[0],1))
X2 =  (X-np.tile(X.min(0),(X.shape[0],1)))/np.tile(X.max(0)-X.min(0),(X.shape[0],1))

def kmeans(sample,K,maxiter):
    N = sample.shape[0] #total sample number
    D = sample.shape[1] #sample dimension
    idx = random.sample(range(N),K) #pick k item from N
    C = sample[idx,:] #K center
    L = np.zeros((N,1)) #group no. of each sample
    dist = np.zeros((N,K))
    iter = 0
    while(iter<maxiter):
        for i in range(K):
            dist[:,i] = np.sum((sample-np.tile(C[i,:],(N,1)))**2,1) 
        L1 = np.argmin(dist,1)
        if(iter>0 and np.array_equal(L,L1)):
            break
        L = L1
        for i in range(K):
            idx = np.nonzero(L==i)[0]
            if(len(idx)>0):
                C[i,:] = np.mean(sample[idx,:],0)
        iter+=1
    wicd = np.mean(np.sqrt(np.sum((sample-C[L,:])**2,1)))
    return C,L,wicd

C,L,wicd = kmeans(X,3,1000)
print("X:        ",wicd)

C,L,wicd = kmeans(X1,3,1000)
print("X1:       ",wicd)


C,L,wicd = kmeans(X2,3,1000)
print("X2:       ",wicd)