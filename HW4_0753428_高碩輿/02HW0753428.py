from sklearn import datasets
import numpy as np
from collections import Counter

def KNN(test,train,target,K):
    N = train.shape[0]
    dist = np.sum((np.tile(test,(N,1))-train)**2,1) 
    idx = sorted(range(len(dist)),key = lambda i:dist[i])[0:K]
    return Counter(target[idx]).most_common(1)[0][0]

iris = datasets.load_iris()
X = iris.data
target = iris.target
N = X.shape[0]

for j in range(1,11):
    CF = np.zeros((3,3))
    for i in range(N):
        train_idx = np.setdiff1d(np.arange(N),i)
        guess = KNN(X[i,:],X[train_idx,:],target[train_idx],j)
        CF[target[i],guess] = CF[target[i],guess]+1
    print(CF)
