import math
import numpy as np
from scipy import signal
import sklearn.metrics as skmetrics
import scipy.io

from initGEQ import initGEQ

def createOutputData():
    print("starting function")
    dataInput = np.loadtxt('data/trainValid/dataInput.csv', delimiter=',')
    dataOutput = np.zeros((len(dataInput),31))
    dataError = np.zeros((len(dataInput),3))
    print("working")

    for i in range((len(dataInput))):
        [numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw] = initGEQ(dataInput[i].reshape(31,1))
        dataOutput[i] = G2opt_db.T
        dataError[i] = ErrorFunction(numsopt,densopt,fs,dataInput[i].reshape(31,1),fc1)
    
    # define data
    data = np.asarray(dataOutput, dtype=np.float64,)
    # save to csv file
    np.savetxt('data/dataOutput.csv', data, delimiter=',')
    
    # define data
    dataErr = np.asarray(dataError, dtype=np.float64,)
    # save to csv file
    np.savetxt('data/dataError.csv', dataErr, delimiter=',')
    
def ErrorFunction(numsopt,densopt,fs,G_db,fc1):
    N_freq = 31
    w = fc1
    H_opt = np.ones((N_freq,31), dtype=complex)
    H_opt_tot = np.ones((N_freq,1), dtype=complex)
    
    for k in range(31):
        w, h = signal.freqz(numsopt[:,k], densopt[:,k],worN=w,fs=fs)
        H_opt[:,k]= h
        H_opt_tot = H_opt[:,[k]]  * H_opt_tot
    
    e1 = 20*np.log10(np.abs(H_opt_tot)).T

    error = np.abs(e1.reshape((31,1))-G_db.reshape((31,1)))
    error_MAE = np.mean(error)     # = MAE
    MSE = skmetrics.mean_squared_error(e1[0], G_db)
    RMSE = np.sqrt(MSE)
    MAE = skmetrics.mean_absolute_error(e1[0], G_db)
    
    return MSE, RMSE, MAE


def Errors():
    
    dataError = np.loadtxt("Data/dataError.csv",delimiter=',').T
  

    maxMSE = np.max(dataError[0])
    maxRMSE = np.max(dataError[1])
    maxMAE = np.max(dataError[2])
   
    minMSE = np.min(dataError[0])
    minRMSE = np.min(dataError[1])
    minMAE = np.min(dataError[2])
    
    return maxMSE,maxRMSE,maxMAE, minMSE, minRMSE, minMAE


#createOutputData()
#print(Errors())
