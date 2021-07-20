import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from groupData import groupData

#train and validData
InputDataLarge = np.loadtxt("data/trainValid/dataInputTrain.csv",delimiter=",")
OutputDataLarge = np.loadtxt ("data/trainValid/dataOutputTrain.csv", delimiter=",")
#testData
InputDataTest = np.loadtxt("data/test/dataInputTest.csv",delimiter=",")
OutputDataTest = np.loadtxt ("data/test/dataOutputTest.csv", delimiter=",")

#transformData
scaler = MinMaxScaler(feature_range=(0, 1))

InputData_transformed = scaler.fit_transform(InputDataTest)
OutputData_transformed = scaler.fit_transform(OutputDataTest)

InputDataL_transformed = scaler.fit_transform(InputDataLarge)
OutputDataL_transformed = scaler.fit_transform(OutputDataLarge)

#GroupData
val_split = 0.2
InputDataTrainL,InputDataValidL,OutputDataTrainL,OutputDataValidL = groupData(InputDataL_transformed,OutputDataL_transformed,val_split)

#chooseModel
modelName="modelOne"

#modelName="modelTwo"

#loadModel
model = tf.keras.models.load_model("models/kerasTunerModels/"+modelName)

print("Evaluation of the model" + modelName)

training = model.evaluate(InputDataTrainL,OutputDataTrainL)
print("training loss, training accuracy:", training)

evaluate = model.evaluate(InputDataValidL,OutputDataValidL)
print("valid loss, valid accuracy:", evaluate)

result = model.evaluate(InputData_transformed,OutputData_transformed)
print("test loss, test accuracy:", result)
