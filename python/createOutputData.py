import numpy as np

from initGEQ import initGEQ
from getErrors import getErrors

def createOutputData():
    dataInput = np.loadtxt('data/test/dataInputTest.csv', delimiter=',')
    dataOutput = np.zeros((len(dataInput),31))
    dataError = np.zeros((len(dataInput),3))
    dataErrorAbsolute = np.zeros((len(dataInput),31))

    for i in range((len(dataInput))):
        [numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw] = initGEQ(dataInput[i].reshape(31,1))
        dataOutput[i] = G2opt_db.T
        helper = getErrors(numsopt,densopt,fs,dataInput[i].reshape(31,1),fc1)
        dataError[i] = helper[0:3]
        dataErrorAbsolute[i] = helper[3:4][0].reshape(1,31)
    
    # define data
    data = np.asarray(dataOutput, dtype=np.float64,)
    # save to csv file
    np.savetxt('data/dataOutputTestNew.csv', data, delimiter=',')
    
    # define data
    dataErr = np.asarray(dataError, dtype=np.float64,)
    dataErrAbsolute = np.asarray(dataErrorAbsolute, dtype=np.float64,)
    # save to csv file
    np.savetxt('data/dataErrorTestNew.csv', dataErr, delimiter=',')
    np.savetxt('data/dataErrorTestNewAbsolute.csv', dataErrAbsolute, delimiter=',')
   
# create output data
#createOutputData()
