"""
Function for initializing the graphic Equalizer

    Parameters
    ----------
    G_db : ndarray 31x1
        Command gains in dB
        
    Returns
    -------
    numsopt : ndarray
        numerator parts of the 31 filters
    densopt : ndarray
        denominator parts of the 31 filters
    fs : float
        sample frequency
    fc2 : ndarray
        center frequencies and additional design points between them
    G_db2 : ndarray
        interpolate target gains linearly b/w command gains
        
    Notes
    -----
    Python reference to Liski, J.; Välimäki, V. The quest for the best graphic equalizer. In Proceedings of the International Conference
    on Digital Audio Effects (DAFx-17), Edinburgh, UK, 5–9 September 2017; pp. 95–102

"""

import numpy as np

from functions.pareq import pareq
from functions.pareqVectorized import pareqVectorized
from functions.interactionMatrix import interactionMatrix

def initGEQFast(G_db,wg,wc,c,bw,leak,fs,fc2,fc1):
    
    bw = bw
    c = c 
    wg = wg 
    wc = wc
    leak = leak 

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
    
    # for k in range(31):
    #     [num,den] = pareq(G2opt[k],G2wopt[k],wg[k],bw[k])
    #     numsopt[:,k] = num
    #     densopt[:,k] = den
    
    numsopt, densopt = pareqVectorized(G2opt.reshape(31),G2wopt.reshape(31),wg,bw)
   
    return numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw
