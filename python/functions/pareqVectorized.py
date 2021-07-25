"""
Second-order parametric equalizing filter desigh with adjustable bandwidth gain, vectorized version

    Parameters
    ----------
    G : float64
        peak gain (linear)
    GB : float64
        bandwidth gain (linear)
    w0: float64
        center frequency (rads/sample)
    B : float64
        bandwidth (rads/sample)
    
    Returns
    -------
    num : ndarray
        numerator coefficients [b0,b1,b2]
    den : ndarray
        denominator coefficients [1,a1,a2]
        
    Notes
    -----
    Python reference to Liski, J.; Välimäki, V. The quest for the best graphic equalizer. In Proceedings of the International Conference
    on Digital Audio Effects (DAFx-17), Edinburgh, UK, 5–9 September 2017; pp. 95–102
    
"""

import numpy as np

def pareqVectorized(G, GB, w0, B):
    
    beta =  np.where(G == 1, np.tan(B/2.), np.multiply(np.sqrt(np.divide(np.abs(GB**2-1),abs(G**2-GB**2))),np.tan(B/2)))

    num = np.array([(1+G*beta)/(1+beta), -2*np.cos(w0)/(1+beta), (1-G*beta)/(1+beta)])
    den = np.array([np.ones(31), -2*np.cos(w0)/(1+beta), (1-beta)/(1+beta)])
    
    return num,den