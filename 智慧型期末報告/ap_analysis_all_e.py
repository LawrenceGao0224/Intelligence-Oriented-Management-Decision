#%reset -f

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation
import supportingfunctions
import time
import matplotlib.pyplot as plt
from itertools import cycle
start = time.time()

AutoscalingFlag = True #True: with autoscaling, False: without autoscaling

OriginalX, Xpd = supportingfunctions.loadunsuperviseddata() #Load data set
# Delete variables with zero variance
Var0Variable = supportingfunctions.variableszerovariance(OriginalX)
if len(Var0Variable[0]) != 0:
    OriginalX = np.delete(OriginalX, Var0Variable, 1)

# 1. Autoscale each variable (if necessary)
if AutoscalingFlag:
    X = (OriginalX - OriginalX.mean(axis=0)) / OriginalX.std(axis=0, ddof=1) #With autoscaling
else:
    X = OriginalX #Without autoscaling

# 2. Decide preference
# Similarity is negative Euclidean distance.
SimilarityMatrix = ( -(X[:, np.newaxis] - X)**2 ).sum(axis=2)
Preference = np.median( SimilarityMatrix )
print(Preference)
 #類似度行列の中央値
#Preference = np.median( SimilarityMatrix ) #類似度行列の中央値

# 3. Decide damping factor
DampingFactor = 0.5

# 4. Run AP
APResults = AffinityPropagation(damping=DampingFactor, preference=Preference)
APResults.fit(X)
ClusterNum = APResults.labels_ + 1
cluster_centers_indices = APResults.cluster_centers_indices_
labels = APResults.labels_

n_clusters_ = len(cluster_centers_indices)
# 5. Visualize clustering result
pca = PCA() #PCA
ScoreT = pca.fit(X).transform(X)
supportingfunctions.makettplotwithclustering(ScoreT, ClusterNum, Xpd, cluster_centers_indices,labels)


# Save result
supportingfunctions.savematrixcsv( ClusterNum, Xpd.index, "ClusterNum")
print('cluster_centers_indices:', (cluster_centers_indices).size)

end = time.time()
elapsed = end - start
print ("Time taken: ", elapsed, "seconds.")
