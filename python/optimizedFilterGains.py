import numpy as np

from interactionMatrix import interactionMatrix

def optimizedFilterGains(G_db,wg,wc,c,bw,leak):

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

    return G2opt_db