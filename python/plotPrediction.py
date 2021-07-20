import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from pareq import pareq

def plotPredictions(filtergainsPrediction,G_db,fs,fc2,fc1,bw,G2opt_db,numsopt,densopt):
    
    G_db2 = np.zeros([61,1])
    G_db2[::2] = G_db
    G_db2[1::2] = (G_db[:len(G_db)-1:1]+G_db[1::1])/2
    wg = 2*math.pi*fc1/fs
    c = 0.38
    
    numsoptPred = np.zeros((3,31))
    densoptPred = np.zeros((3,31))
    G = 10 **(filtergainsPrediction/20)
    Gw_db = c * filtergainsPrediction
    Gw = 10 **(Gw_db/20)
    
    for k in range(31):
        [num,den] = pareq(G[k],Gw[k],wg[k],bw[k])
        numsoptPred[:,k] = num
        densoptPred[:,k] = den

    N_freq = 2 **12
    w = np.logspace(np.log10(9),np.log10(22050), N_freq)
    H_optPred = np.ones((N_freq,31), dtype=complex)
    H_opt_totPred = np.ones((N_freq,1), dtype=complex)
    
    for k in range(31):
        w, h = signal.freqz(numsoptPred[:,k], densoptPred[:,k],worN=w,fs=fs)
        H_optPred[:,k]= h
        H_opt_totPred = H_optPred[:,[k]]  * H_opt_totPred
        
    H_opt = np.ones((N_freq,31), dtype=complex)
    H_opt_tot = np.ones((N_freq,1), dtype=complex)
    
    numsopt = numsopt
    densopt = densopt
    
    for k in range(31):
        w, h = signal.freqz(numsopt[:,k], densopt[:,k],worN=w,fs=fs)
        H_opt[:,k]= h
        H_opt_tot= H_opt[:,[k]]  * H_opt_tot
    
    fig = plt.figure(4)
    
    plt.semilogx(w,20*np.log10(np.abs(H_opt_tot)), linewidth=3.0) 
    plt.semilogx(w,20*np.log10(np.abs(H_opt_totPred)), color="orange") #predicted 
    
    #plt.semilogx(w,20*np.log10(np.abs(H_opt)))
    plt.plot(fc2,G_db2, "ro", markersize=4, markerfacecolor="none")
    plt.plot(fc1,filtergainsPrediction, "ro", markersize=6, markerfacecolor="none",marker="x", markeredgecolor="r")
    plt.plot(fc1,G2opt_db, "ro", markersize=6, markerfacecolor="none", marker="s",markeredgecolor="r")
    plt.ylabel("Pegel in dB")
    plt.xlabel("Frequenz in Hz")
    #plt.title("Predicted frequency response")
    plt.xticks([10, 30, 100, 1000, 3000, 10000])
    plt.yticks(np.arange(-15,20,5))
    plt.grid(which="both", linestyle="--", color="grey")
    filtergain = mlines.Line2D([], [], linestyle='None',
                          markersize=8, markerfacecolor="none", marker="o", markeredgecolor="r", label="Target Gains")
    targetgain = mlines.Line2D([], [], linestyle='None',
                          markersize=8, markerfacecolor="none", marker="s", markeredgecolor="r", label="Optimized filter gains")
    filterpredicted = mlines.Line2D([], [], linestyle='None',
                          markersize=8, markerfacecolor="none", marker="x", markeredgecolor="r", label="Predicted filter gains")
    frequencyCalculated = mlines.Line2D([], [], linestyle='None',
                          markersize=8, markerfacecolor="none", marker="_", markeredgecolor="b", label="OGEQ")
    frequencyPredicted = mlines.Line2D([], [], linestyle='None',
                          markersize=8, markerfacecolor="none", marker="_", markeredgecolor="orange", label="NGEQ")                      
    plt.legend(handles=[filtergain,targetgain,filterpredicted,frequencyCalculated,frequencyPredicted])
    #plt.show()
    
    return fig