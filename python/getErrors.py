import numpy as np
import sklearn.metrics as skmetrics
from scipy import signal

def getErrors(numsopt,densopt,fs,G_db,fc1):
    N_freq = 31
    w = fc1
    H_opt = np.ones((N_freq,31), dtype=complex)
    H_opt_tot = np.ones((N_freq,1), dtype=complex)
    
    for k in range(31):
        w, h = signal.freqz(numsopt[:,k], densopt[:,k],worN=w,fs=fs)
        H_opt[:,k]= h
        H_opt_tot = H_opt[:,[k]]  * H_opt_tot
    
    e1 = 20*np.log10(np.abs(H_opt_tot)).T
    errorAbsolute = np.abs(e1.reshape((31,1))-G_db.reshape((31,1))).reshape((1,31))
    error_MAE = np.mean(errorAbsolute)     # = MAE
    MSE = skmetrics.mean_squared_error(e1[0], G_db)
    RMSE = np.sqrt(MSE)
    MAE = skmetrics.mean_absolute_error(e1[0], G_db)
    
    return MSE, RMSE, MAE, errorAbsolute

