import math
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
feature = iris.data
target = iris.target
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

def T(a,b):  # target :a ,target :b
    data = iris.data
    feature = data[trainidx]
    target = iris.target[trainidx]
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
            
        elif((sum(target[idx]==1))==len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=1
           
        elif((sum(target[idx]==2))==len(idx)*2):
            Tree[t]['leaf']=1
            Tree[t]['decision']=2
           
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
                    thisIG = IG(sum(target[G1]==b),sum(target[G1]==a),sum(target[G2]==b),sum(target[G2]==a))
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
                if(sum(target[idx]==b)>sum(target[idx]==a)):
                    Tree[t]['decision']=b
                else:
                    Tree[t]['decision']=a
        t = t+1  
    return Tree

Y=0
N=0
group = np.mod(np.arange(150),50)//5
for testg in range(10):
    trainidx = np.where(group!=testg)[0]
    testidx = np.where(group==testg)[0]
    t1 = T(0,1)
    t2 = T(0,2)
    t3 = T(1,2)
    
    for i in testidx:
        a=0 #0
        b=0 #1
        c=0 #2
        test_feature = feature[i]
        
        now = 0
        while(t1[now]['leaf']==0):
            bestf = t1[now]['selectf']
            thres = t1[now]['threshold']
            if(test_feature[bestf]<=thres):
                now = t1[now]['child'][0]
            else:
                now = t1[now]['child'][1]
        if (t1[now]['decision'])==0:
            a += 1
        else:
            b += 1

        now = 0
        while(t2[now]['leaf']==0):
            bestf = t2[now]['selectf']
            thres = t2[now]['threshold']
            if(test_feature[bestf]<=thres):
                now = t2[now]['child'][0]
            else:
                now = t2[now]['child'][1]
        if (t2[now]['decision']) == 0:
            a += 1
        else:
            c += 1
            
        now = 0
        while(t3[now]['leaf']==0):
            bestf = t3[now]['selectf']
            thres = t3[now]['threshold']
            if(test_feature[bestf]<=thres):
                now = t3[now]['child'][0]
            else:
                now = t3[now]['child'][1]       
        if (t3[now]['decision']) == 1:
            b += 1
        else:
            c += 1

        if(a==2 and 0==target[i]):
            Y += 1
                #print(0)
        elif(b==2 and 1==target[i]):
            Y += 1
                #print(1)
        elif(c==2 and 2==target[i]):
            Y += 1
                #print(2)
        elif(a==1 and b ==1 and c == 1 and target[i] == 0):
            Y += 1
        else:
            N += 1
print("錯誤率: ", N/150)
   
    
    