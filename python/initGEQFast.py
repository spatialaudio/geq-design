import numpy as np

from pareq import pareq
from interactionMatrix import interactionMatrix

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
    
    for k in range(31):
        [num,den] = pareq(G2opt[k],G2wopt[k],wg[k],bw[k])
        numsopt[:,k] = num
        densopt[:,k] = den
   
    return numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw
