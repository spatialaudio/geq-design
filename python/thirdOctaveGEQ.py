import numpy as np
from initGEQ import initGEQ


def thirdOctaveGEQ(commandGains):
    
    G_db = commandGains.reshape((31,1))
    [numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw] = initGEQ(G_db) 
    #print(G2opt_db.reshape(1,31))
    
    return 


#example commandGain
#commandGain = 12* np.ones(31)
commandGain= np.append(np.tile([[12,12,-12]],10),12).reshape(31,1)
#run GEQ
thirdOctaveGEQ(commandGain)