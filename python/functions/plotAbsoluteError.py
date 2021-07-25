"""
Evaluate and plot absolute Error

    Parameters
    ----------
    fc1 : ndarray
        center frequences
    input: int
        integer for i-th setting in datasets
    
        
    Returns
    -------
    absolute Error of GEQ and NGEQ as a figure
    
    Notes
    -----
    
"""


import matplotlib.pyplot as plt
import numpy as np

def plotAbsoluteError(fc1,input):
    
    
    w = fc1
    input = input
    errorAbs = np.loadtxt("data/test/dataErrorTestAbsolute.csv",delimiter=",")[input:input+1].reshape(31)
    errorAbsPred = np.loadtxt("data/prediction/predictionErrorTestAbsolute.csv",delimiter=",")[input:input+1].reshape(31)

    error = np.loadtxt("data/test/dataErrorTest.csv",delimiter=",")[input:input+1].reshape(1,3)
    errorPred = np.loadtxt("data/prediction/predictionErrorTest.csv",delimiter=",")[input:input+1]
    
    print("MSE, RMSE, MAE;" ,error)
    print("predicted MSE, RMSE, MAE;" ,errorPred)


    fig = plt.figure(3)

    plt.semilogx(w,errorAbs, "s",markerfacecolor="none")
    plt.semilogx(w,errorAbsPred, color="orange") #predicted
    plt.ylabel("Fehler in dB")
    plt.xlabel("Frequenz in Hz")
    plt.title("Absolute Error")
    plt.xticks([10, 30, 100, 1000, 3000, 10000])
    plt.yticks(np.arange(0,1.2,0.2))
    plt.grid(which="both", linestyle="--", color="grey")
    
    #plt.show()
    return fig