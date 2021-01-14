#Supporting functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
import math
from itertools import cycle
#load unsupervised dataset
def loadunsuperviseddata():
    OriginalXpd = pd.read_csv("SF.csv", encoding='SHIFT-JIS', index_col=0)
    OriginalX = OriginalXpd.as_matrix()
    OriginalX = OriginalX.astype(float)
    return (OriginalX, OriginalXpd)
         
#Find variables with zero variance
def variableszerovariance( X ):
    Var0Variable = np.where( X.var(axis=0) == 0 )
    if len(Var0Variable[0]) == 0:
        print( "No variables with zero variance" )
    else:
        print( "{0} variable(s) with zero variance".format(len(Var0Variable[0])))
        print( "Variable number: {0}".format(Var0Variable[0]+1) )
        print( "The variable(s) is(are) deleted." )
    return Var0Variable

#Save matrix
def savematrixcsv( X, index, filename):
    Xpd = pd.DataFrame(X)
    Xpd.index = index
    exec("Xpd.to_csv( \"{}.csv\", header = False )".format( filename ) )


    
#make tt plots for PCA with clustering result
#def makettplotwithclustering(ScoreT, ClusterNum, Xpd, cluster_centers_indices):
def makettplotwithclustering(ScoreT, ClusterNum, Xpd, cluster_centers_indices,labels):   
    #plt.scatter(ScoreT[:,0], ScoreT[:,1],  c=ClusterNum, cmap=plt.get_cmap('jet'))
    #for numofsample in np.arange( 0, ScoreT.shape[0]-1):
        #plt.text(ScoreT[numofsample,0], ScoreT[numofsample,1],Xpd.index[numofsample], horizontalalignment='left', verticalalignment='top')
    ''' 
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range((cluster_centers_indices).size), colors):
        print(k, col)
        class_members = labels == k
        cluster_center = ScoreT[cluster_centers_indices[k],0:2]
        print(cluster_center)
        plt.plot(ScoreT[class_members, 0], ScoreT[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    '''

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range((cluster_centers_indices).size), colors):
        class_members = labels == k
        cluster_center = ScoreT[cluster_centers_indices[k],0:2]
        #print(cluster_center)
        plt.plot(ScoreT[class_members, 0], ScoreT[class_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)   
        for x in ScoreT[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()



