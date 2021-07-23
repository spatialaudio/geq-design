"""
get different errors of a explicit dataset

    Parameters
    ----------
    dataName: string
        data set whose error is to be evaluated 
        
    Returns
    -------
    different Erros (MSE,MAE,AE) of a given dataset
    Notes
    -----
    
"""

import numpy as np

dataNamePredict = "prediction/predictionErrorTest"
dataName = "test/dataErrorTest"

def Errors(dataName):
    
    dataError = np.loadtxt("data/" + dataName + ".csv" ,delimiter=',').T
  
    maxMSE = np.max(dataError[0])
    maxRMSE = np.max(dataError[1])
    maxMAE = np.max(dataError[2])
   
    minMSE = np.min(dataError[0])
    minRMSE = np.min(dataError[1])
    minMAE = np.min(dataError[2])


    print(dataName + ": \n maxMSE:" + str(maxMSE) + "\n max MAE:" + str(maxMAE) + "\n min MSE:" + str(minMSE) + "\n min MAE:" + str(minMAE))

    dataErrorAbsolute = np.loadtxt("data/" + dataName + "Absolute.csv",delimiter=',').T
    
    
    meanMSE = np.mean(dataError[0])
    print("mean MSE:" + str(meanMSE))
    meanMAE = np.mean(dataError[2])
    print("mean MAE:" + str(meanMAE))

    maxAbsolute = np.max(dataErrorAbsolute)
    print("Max absolute Error:" + str(maxAbsolute))
    minAbsolute = np.min(dataErrorAbsolute)
    print("Min absolute Error:" + str(minAbsolute) + "\n")

    return maxMSE,maxRMSE,maxMAE, minMSE, minRMSE, minMAE

Errors(dataName)
Errors(dataNamePredict)