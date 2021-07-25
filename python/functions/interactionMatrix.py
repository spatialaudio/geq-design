"""
Compute the interaction matrix of a cascade GEQ

    Parameters
    ----------
    G : ndarray
        linear gain at which the leakage is determined    
    c : float
        gain factor at bandwidth (0.5 refers to db(G)/2)
    wg : ndarray
        command frequencies i.e. filter center frequencies (in rad/sample)
    wc : ndarray
        design frequencies (rad/sample) at which leakage is computed
    bw : ndarray 
        bandwidth of filters in radians
        
    Returns
    -------
    leak : ndarray
        N by M matrix showing how the magnitude responses of the band filters leak to the design frequencies.
        N is determined from the length of the array wc (number of design frequencies) whereas M is 
        determined from the length of wg (number of filters)
        
    Notes
    -----
    Python reference to Liski, J.; Välimäki, V. The quest for the best graphic equalizer. In Proceedings of the International Conference
    on Digital Audio Effects (DAFx-17), Edinburgh, UK, 5–9 September 2017; pp. 95–102
"""

import numpy as np
from scipy import signal 

from functions.pareq import pareq

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