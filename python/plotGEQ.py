"""
Evaluate and plot

    Parameters
    ----------
    numsopt : ndarray
        numerator coefficients for each filter
    densopt : ndarray
        denominator coefficients for each filter
    fs : float
        sample frequency
    fc2 : ndarray
        center frequencies and additional design points between them
    G_db2 : ndarray
        interpolate target gains linearly b/w command gains
        
    Returns
    -------
    
    Notes
    -----
    Python reference to Liski, J.; Välimäki, V. The quest for the best graphic equalizer. In Proceedings of the International Conference
    on Digital Audio Effects (DAFx-17), Edinburgh, UK, 5–9 September 2017; pp. 95–102
    
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def plot(numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1):
    N_freq = 2 **12
    w = np.logspace(np.log10(9),np.log10(22050), N_freq)
    H_opt = np.ones((N_freq,31), dtype=complex)
    H_opt_tot = np.ones((N_freq,1), dtype=complex)
    
    for k in range(31):
        w, h = signal.freqz(numsopt[:,k], densopt[:,k],worN=w,fs=fs)
        H_opt[:,k]= h
        H_opt_tot = H_opt[:,[k]]  * H_opt_tot
    
    
    
    fig = plt.figure(200)
    plt.semilogx(w,20*np.log10(np.abs(H_opt_tot)))
    #plt.semilogx(w,20*np.log10(np.abs(H_opt)))
    plt.plot(fc2,G_db2, "ro", markersize=3, markerfacecolor="none",)
    plt.plot(fc1,G2opt_db, "ro", markersize=4, markerfacecolor="none",marker="x")
    plt.ylabel("Pegel in dB")
    plt.xlabel("Frequenz in Hz")
    #plt.title("Optimized frequency response")
    plt.xticks([10, 30, 100, 1000, 3000, 10000])
    plt.yticks(np.arange(0,15,5))
    plt.grid(which="both", linestyle="--", color="grey")
    filtergain = mlines.Line2D([], [], linestyle='None',
                          markersize=8, markerfacecolor="none", marker="x", markeredgecolor="r", label="Filter gains")
    targetgain = mlines.Line2D([], [], linestyle='None',
                          markersize=8, markerfacecolor="none", marker="o", markeredgecolor="r", label="Command Gains")
    plt.legend(handles=[filtergain,targetgain])
    #plt.show()
    
    
    #fig.savefig("Figures/Opt_alter12dB.pdf")
    
    return fig