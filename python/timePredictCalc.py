import math
import numpy as np
from scipy import signal
import sklearn.metrics as skmetrics
import scipy.io
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from pareq import pareq
from pareqVectorized import pareqVectorized
from interactionMatrix import interactionMatrix
from optimizedFilterGains import optimizedFilterGains
from initGEQFast import initGEQFast

modelName = "kerasTunerModels/modelTwo"
model = tf.keras.models.load_model("models/"+modelName)


def timeCalculation():
    InputDataTest = np.loadtxt('data/test/dataInputTest.csv', delimiter=',')
    tik = time.time()
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
  
    leak = interactionMatrix(10**(17/20)*np.ones(31),c,wg,wc,bw)

    for i in range((len(InputDataTest))):
        [numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw] = initGEQFast(InputDataTest[i].reshape(31,1),wg,wc,c,bw,leak,fs,fc2,fc1)
    
    tok = time.time()
    
    return (tok - tik)* 1000

def timeCalculationFilterGains():
    
    InputDataTest = np.loadtxt('data/test/dataInputTest.csv', delimiter=',')
   
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
  
    leak = interactionMatrix(10**(17/20)*np.ones(31),c,wg,wc,bw)

    
    tik = time.time()
    for i in range((len(InputDataTest))):
        G_db2 = np.zeros([61,1])
        G_db2[::2] = InputDataTest[i].reshape(31,1)
        G_db2[1::2] = (InputDataTest[i].reshape(31,1)[:len(InputDataTest[i].reshape(31,1))-1:1]+InputDataTest[i].reshape(31,1)[1::1])/2

        Gopt_db = np.linalg.lstsq(leak.conj().T, G_db2)[0]
        Gopt = 10**(Gopt_db/20)
    
        leak2 = interactionMatrix(Gopt,c,wg,wc,bw)
        G2opt_db = np.linalg.lstsq(leak2.conj().T, G_db2)[0] #filter gains
    tok = time.time()
    
    return (tok - tik)* 1000


def timePrediction():
    
    InputDataTest = np.loadtxt('data/test/dataInputTest.csv', delimiter=',')
    
    tik = time.time()
    scaler = MinMaxScaler(feature_range=(0, 1))
    InputDataTest_transformed = scaler.fit_transform(InputDataTest) 

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
    
    predictions = scaler.inverse_transform(model(InputDataTest_transformed))
    
    for i in range((len(InputDataTest_transformed))):
        
        numsoptPred = np.zeros((3,31))
        densoptPred = np.zeros((3,31))

        G2opt_db = predictions[i:i+1].reshape(31,1)
        G2opt = 10 **(G2opt_db/20)
        G2wopt_db = 0.38 * G2opt_db
        G2wopt = 10 **(G2wopt_db/20)
    
        for k in range(31):
            [num,den] = pareq(G2opt[k],G2wopt[k],wg[k],bw[k])
            numsoptPred[:,k] = num
            densoptPred[:,k] = den
        
        #numsoptPred, densoptPred = pareqVectorized(G2opt.reshape(31),G2wopt.reshape(31),wg,bw)

    tok = time.time()

    return (tok-tik)*1000

def timePredictionFilterGains():
    
    InputDataTest = np.loadtxt('data/test/dataInputTest.csv', delimiter=',')
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    InputDataTest_transformed = scaler.fit_transform(InputDataTest) 

    tik1 = time.time()
    predictions = scaler.inverse_transform(model(InputDataTest_transformed))
    tok1 = time.time()
 

    return (tok1-tik1)*1000

iterations = 100
timeCalc = 0 
timeCalcFilterGains = 0
timePred = 0
timePredFilterGains = 0

for i in range(iterations):
    [timeCalc,timeCalcFilterGains,timePred,timePredFilterGains] = timeCalculation() + timeCalc,timeCalculationFilterGains()+ timeCalcFilterGains,timePrediction()+timePred,timePredictionFilterGains()+timePredFilterGains

print("Time in ms for" , iterations , "Dataset-Iterations:")
print("Time calc for Dataset in ms:", timeCalc)
print("Time calc filter gains:", timeCalcFilterGains)
print("Time predicted", timePred)
print("Time predicted filter gains", timePredFilterGains)

print("\n Time in ms for 1 Dataset:")
print("Time calc for Dataset in ms:", timeCalc/iterations)
print("Time calc filter gains:", timeCalcFilterGains/iterations)
print("Time predicted", timePred/iterations)
print("Time predicted filter gains", timePredFilterGains/iterations)

print("\n Time in ms for 1 CommandGain Setting:")
print("Time calc for Dataset in ms:", timeCalc/(iterations*1000))
print("Time calc filter gains:", timeCalcFilterGains/(iterations*1000))
print("Time predicted", timePred/(iterations*1000))
print("Time predicted filter gains", timePredFilterGains/(iterations*1000))

