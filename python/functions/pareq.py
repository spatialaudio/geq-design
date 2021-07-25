"""
Second-order parametric equalizing filter desigh with adjustable bandwidth gain

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
import math

def pareq(G, GB, w0, B):
    if G == 1:
        beta = math.tan(B/2.)
    else: 
        beta = np.sqrt(abs(GB**2-1)/abs(G**2-GB**2))*math.tan(B/2)

    num = np.array([(1+G*beta), -2*math.cos(w0), (1-G*beta)]/(1+beta))
    den = np.array([1, -2*math.cos(w0)/(1+beta), (1-beta)/(1+beta)])
    
    return num, den