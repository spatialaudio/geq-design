"""
Plot the predicted frequency response and absolute error of a GEQ

    Parameters
    ----------
    input: int
        select command gain configuration from the input dataset

        
    Returns
    -------
    plot of predicted and calculated frequency response. plot of absoulute error in dB
        
    Notes
    -----
"""



import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from pareq import pareq
from initGEQ import initGEQ
from plotPrediction import plotPredictions
from plotAbsoluteError import plotAbsoluteError
from plotGEQ import plot


#choose input configuration
input = 868

#chooseModel
#modelName = "kerasTunerModels/modelOne"
modelName = "kerasTunerModels/modelTwo"

#loadModel
model = tf.keras.models.load_model("models/"+modelName)

#import dataset
InputData = np.loadtxt("data/test/dataInputTest.csv",delimiter=",")
OutputData = np.loadtxt ("data/test/dataOutputTest.csv", delimiter=",")

#scale data
scaler = MinMaxScaler(feature_range=(0, 1))

InputData_scaled = scaler.fit_transform(InputData) 
OutputData_scaled = scaler.fit_transform(OutputData) 


prediction = model(InputData_scaled[input:input+1])
[numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw] = initGEQ(InputData[input:input+1].reshape(31,1))

print("InputData in db", InputData[input:input+1])
print("InputData scaled", InputData_scaled[input:input+1])
print("Predictions:", prediction)
print("Output scaled:", OutputData_scaled[input:input+1])
print("Predictions in dB", scaler.inverse_transform(prediction))
print("Output in dB", G2opt_db.reshape(1,31))
print("Diff in dB", scaler.inverse_transform(prediction)- G2opt_db.reshape(1,31))




def thirdOctaveGEQwithPredictions(commandGains,filterGainsPrediction):
    
    G_db = commandGains
    filterGains = filterGainsPrediction
    [numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1,bw] = initGEQ(G_db.reshape(31,1))
    fig_predict = plotPredictions(filterGains,G_db,fs,fc2,fc1,bw,G2opt_db, numsopt,densopt)
    fig_calc = plot(numsopt,densopt,fs,fc2,G_db2,G2opt_db,fc1)
    fig_error = plotAbsoluteError(fc1,input)
    
    plt.show()
    
    return



commandGains = InputData[input:input+1].reshape(31,1)
filterGainsPredicted = scaler.inverse_transform(prediction).reshape((31,1))

thirdOctaveGEQwithPredictions(commandGains, filterGainsPredicted)