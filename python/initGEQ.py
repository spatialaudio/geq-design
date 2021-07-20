import math
import numpy as np
from scipy import signal
import sklearn.metrics as skmetrics
import scipy.io
import time

from pareq import pareq
from interactionMatrix import interactionMatrix

def initGEQ(G_db):
    fs = 44.1e3
    fc1 = np.array([19.69,24.80,31.25,39.37,49.61,62.50,78.75,99.21,125.0,157.5,198.4,
    250.0,315.0,396.9,500.0,630.0,793.7,1000,1260,1587,2000,2520,3175,4000,
    5040,6350,8000,10080,12700,16000,20160])
    fc2 = np.zeros(61)
    fc2[::2] = fc1
    fc2[1::2] = np.sqrt(fc1[0:len(fc1)-1:1] * fc1[1::1])
    wg = 2*math.pi*fc1/fs
    wc = 2*math.pi*fc2/fs
    c= 0.38
    bw = np.array((2 ** (1/3) - 2 ** (-1/3)) * wg)
    bw_adjust = 2*math.pi/fs
    bw[::] =[9.178*bw_adjust, 11.56*bw_adjust, 14.57*bw_adjust, 18.36*bw_adjust, 23.13*bw_adjust, 29.14*bw_adjust, 36.71*bw_adjust, 46.25*bw_adjust, 58.28*bw_adjust, 73.43*bw_adjust, 
    92.51*bw_adjust, 116.6*bw_adjust, 146.9*bw_adjust, 185.0*bw_adjust, 233.1*bw_adjust, 293.7*bw_adjust,370*bw_adjust, 466.2*bw_adjust, 587.4*bw_adjust, 740.1*bw_adjust, 932.4*bw_adjust, 1175*bw_adjust, 1480*bw_adjust, 1865*bw_adjust, 2350*bw_adjust, 2846*bw_adjust, 3502*bw_adjust, 4253*bw_adjust, 5038*bw_adjust, 5689*bw_adjust, 5570*bw_adjust]
    leak = interactionMatrix(10**(17/20)*np.ones(31),c,wg,wc,bw)
    

    G_db2 = np.zeros([61,1])
    G_db2[::2] = G_db
    G_db2[1::2] = (G_db[:len(G_db)-1:1]+G_db[1::1])/2

    Gopt_db = np.linalg.lstsq(leak.conj().T, G_db2)[0]
    Gopt = 10**(Gopt_db/20)
    
    leak2 = interactionMatrix(Gopt,c,wg,wc,bw)
    G2opt_db = np.linalg.lstsq(leak2.conj().T, G_db2)[0] #filter gains
    G2opt = 10 **(G2opt_db/20)
    G2wopt_db = c * G2opt_db
    G2wopt = 10 **(G2wopt_db/20)
    
    numsopt = np.zeros((3,31))
    densopt = np.zeros((3,31))
    
    for k in range(31):
        [num,den] = pareq(G2opt[k],G2wopt[k],wg[k],bw[k])
        numsopt[:,k] = num
        densopt[:,k] = den
   
    return numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw


