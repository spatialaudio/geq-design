"""
calculate

    Parameters
    ----------
    commandGains: ndarray
        user set command gains 
        
    Returns
    -------
    optimized filter gains of GEQ
    
    Notes
    -----
"""



import numpy as np
from initGEQ import initGEQ

def thirdOctaveGEQ(commandGains):
    
    G_db = commandGains.reshape((31,1))
    [numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw] = initGEQ(G_db) 
    print(G2opt_db.reshape(1,31))
    
    return 


#example commandGain
commandGain = 12* np.ones(31)
#run GEQ
thirdOctaveGEQ(commandGain)