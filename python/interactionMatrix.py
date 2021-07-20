import numpy as np
from scipy import signal 

from pareq import pareq

def interactionMatrix(G,c,wg,wc,bw):

    M = len(wg)
    N = len(wc)
    leak = np.zeros([M,N]) 
    Gdb = 20 * np.log10(G)
    Gdbw = c * Gdb
    Gw = 10 ** (Gdbw/20)
    
    for m in range(M):
        [num, den] = pareq(G[m],Gw[m],wg[m],bw[m])
        w,h = signal.freqz(num, den, wc)
        Gain = 20*np.log10(np.abs(h))/Gdb[m]
        leak[m,:] = np.abs(Gain)
        
    return leak