import numpy as np
import tensorflow as tf
import time
from sklearn.preprocessing import MinMaxScaler

modelName = "kerasTunerModels/modelTwo"
model = tf.keras.models.load_model("models/"+modelName)

InputDataTest = np.loadtxt("data/test/dataInputTest.csv",delimiter=",")
OutputDataTest = np.loadtxt("data/test/dataOutputTest.csv",delimiter=",")


#TransformData
scaler = MinMaxScaler(feature_range=(0, 1))
#only for test
InputData_transformed = scaler.fit_transform(InputDataTest)
OutputData_transformed = scaler.fit_transform(OutputDataTest)

predictions = scaler.inverse_transform(model(InputData_transformed))
predictionSave = np.asarray(predictions, dtype=np.float64,)
np.savetxt('data/prediction/predictedOutputTestNew.csv', predictionSave, delimiter=',')

