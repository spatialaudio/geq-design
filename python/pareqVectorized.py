import numpy as np

def pareqVectorized(G, GB, w0, B):
    
    beta =  np.where(G == 1, np.tan(B/2.), np.multiply(np.sqrt(np.divide(np.abs(GB**2-1),abs(G**2-GB**2))),np.tan(B/2)))

    num = np.array([(1+G*beta)/(1+beta), -2*np.cos(w0)/(1+beta), (1-G*beta)/(1+beta)])
    den = np.array([np.ones(31), -2*np.cos(w0)/(1+beta), (1-beta)/(1+beta)])
    
    return num,den