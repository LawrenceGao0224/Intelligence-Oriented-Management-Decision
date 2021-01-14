import matplotlib.pyplot as plt
import numpy as np
call = {10600:418,10700:332,10800:245,10900:173, 11000:115,11100:71,11200:40.5}
put = {10600:24.5,10700:37,10800:57,10900:86,11000:127,11100:183,11200:251}
x = np.arange(10500,11501)

def callr(K):#買權
    global x
    global call
    return np.maximum(x-K,0)-call[K]
def putr(K):#賣權
    global x
    global call
    return np.maximum(K-x,0)-put[K]

y1 = callr(10800)
y2 = callr(10900)
y3 = callr(11000)
y4 = y1-y2
y5 = y2-y3
y6 = y1-y3


plt.plot(x,y4,'r',x,y5,'g',x,y6,'b',[x[0],x[-1]],[0,0],'--k')