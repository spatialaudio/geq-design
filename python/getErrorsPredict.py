import math
import numpy as np

from initGEQ import initGEQ
from pareq import pareq
from getErrors import getErrors

def createPredictionError():
    dataInput = np.loadtxt('data/test/dataInputTest.csv', delimiter=',')
    dataPrediction = np.loadtxt("data/prediction/predictedOutputTest.csv", delimiter=",")

    dataError = np.zeros((len(dataInput),3))
    dataErrorAbsolute = np.zeros((len(dataInput),31))

    fs = 44.1e3
    fc1 = np.array([19.69,24.80,31.25,39.37,49.61,62.50,78.75,99.21,125.0,157.5,198.4,
    250.0,315.0,396.9,500.0,630.0,793.7,1000,1260,1587,2000,2520,3175,4000,
    5040,6350,8000,10080,12700,16000,20160])
    fc2 = np.zeros(61)
    fc2[::2] = fc1
    fc2[1::2] = np.sqrt(fc1[0:len(fc1)-1:1] * fc1[1::1])
    wg = 2*math.pi*fc1/fs
    wc = 2*math.pi*fc2/fs
    c= 0.38
    bw = np.array((2 ** (1/3) - 2 ** (-1/3)) * wg)
    bw_adjust = 2*math.pi/fs
    bw[::] =[9.178*bw_adjust, 11.56*bw_adjust, 14.57*bw_adjust, 18.36*bw_adjust, 23.13*bw_adjust, 29.14*bw_adjust, 36.71*bw_adjust, 46.25*bw_adjust, 58.28*bw_adjust, 73.43*bw_adjust, 
    92.51*bw_adjust, 116.6*bw_adjust, 146.9*bw_adjust, 185.0*bw_adjust, 233.1*bw_adjust, 293.7*bw_adjust,370*bw_adjust, 466.2*bw_adjust, 587.4*bw_adjust, 740.1*bw_adjust, 932.4*bw_adjust, 1175*bw_adjust, 1480*bw_adjust, 1865*bw_adjust, 2350*bw_adjust, 2846*bw_adjust, 3502*bw_adjust, 4253*bw_adjust, 5038*bw_adjust, 5689*bw_adjust, 5570*bw_adjust]
    
    for i in range((len(dataInput))):
        
        numsoptPred = np.zeros((3,31))
        densoptPred = np.zeros((3,31))

        G2opt_db = dataPrediction[i:i+1].reshape(31,1)
        G2opt = 10 **(G2opt_db/20)
        G2wopt_db = 0.38 * G2opt_db
        G2wopt = 10 **(G2wopt_db/20)
    
        for k in range(31):
            [num,den] = pareq(G2opt[k],G2wopt[k],wg[k],bw[k])
            numsoptPred[:,k] = num
            densoptPred[:,k] = den

        helper = getErrors(numsoptPred,densoptPred,fs,dataInput[i].reshape(31,1),fc1)
        dataError[i] = helper[0:3]
        dataErrorAbsolute[i] = helper[3:4][0].reshape(1,31)

    # define data
    dataErr = np.asarray(dataError, dtype=np.float64,)
    dataErrAbsolute = np.asarray(dataErrorAbsolute, dtype=np.float64,)
    # save to csv file
    np.savetxt('data/prediction/predictionErrorTestNew.csv', dataErr, delimiter=',')
    np.savetxt('data/prediction/predictionErrorTestNewAbsolute.csv', dataErrAbsolute, delimiter=',')
    
#createPredictionError()

