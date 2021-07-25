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
import matplotlib.pyplot as plt
from initGEQ import initGEQ
from functions.plotGEQ import plot

def thirdOctaveGEQ(commandGains):
    
    G_db = commandGains.reshape((31,1))
    [numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw] = initGEQ(G_db) 
    fig_calc = plot(numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1)
    plt.show()
    
    return 


#example commandGain
commandGain = 12* np.ones(31)
#run GEQ
thirdOctaveGEQ(commandGain)