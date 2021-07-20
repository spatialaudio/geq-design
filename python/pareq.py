import math
import numpy as np


def pareq(G, GB, w0, B):
    if G == 1:
        beta = math.tan(B/2.)
    else: 
        beta = np.sqrt(abs(GB**2-1)/abs(G**2-GB**2))*math.tan(B/2)

    num = np.array([(1+G*beta), -2*math.cos(w0), (1-G*beta)]/(1+beta))
    den = np.array([1, -2*math.cos(w0)/(1+beta), (1-beta)/(1+beta)])
    
    return num, den