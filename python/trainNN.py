"""
train a model

    Parameters
    ----------
    modelName: string
        model what is to be trained 
    val_split: float
        select how large the validation data portion is    
        
    Returns
    -------
    trained model 

    Notes
    -----
    if the trained model is to be saved, comment out the last line
"""


import numpy as np
import tensorflow as tf
import datetime
import time
from sklearn.preprocessing import MinMaxScaler

from groupData import groupData

#GetData
#only for test
InputDataTest = np.loadtxt("data/test/dataInputTest.csv",delimiter=",")
OutputDataTest = np.loadtxt ("data/test/dataOutputTest.csv", delimiter=",")
#for training
InputDataSmall = np.loadtxt("data/trainValid/dataInputSmall.csv",delimiter=",")
OutputDataSmall = np.loadtxt ("data/trainValid/dataOutputSmall.csv", delimiter=",")
InputDataMid = np.loadtxt("data/trainValid/dataInputMid.csv",delimiter=",")
OutputDataMid = np.loadtxt ("data/trainValid/dataOutputMid.csv", delimiter=",")
InputDataLarge = np.loadtxt("data/trainValid/dataInputTrain.csv",delimiter=",")
OutputDataLarge = np.loadtxt ("data/trainValid/dataOutputTrain.csv", delimiter=",")


#TransformData
scaler = MinMaxScaler(feature_range=(0, 1))
#only for test
InputData_transformed = scaler.fit_transform(InputDataTest)
OutputData_transformed = scaler.fit_transform(OutputDataTest)
#for training
InputDataS_transformed = scaler.fit_transform(InputDataSmall)
OutputDataS_transformed = scaler.fit_transform(OutputDataSmall)
InputDataM_transformed = scaler.fit_transform(InputDataMid)
OutputDataM_transformed = scaler.fit_transform(OutputDataMid)
InputDataL_transformed = scaler.fit_transform(InputDataLarge)
OutputDataL_transformed = scaler.fit_transform(OutputDataLarge)


#GroupData
val_split = 0.2
InputDataTrainS,InputDataValidS,OutputDataTrainS,OutputDataValidS = groupData(InputDataS_transformed,OutputDataS_transformed,val_split)
InputDataTrainM,InputDataValidM,OutputDataTrainM,OutputDataValidM = groupData(InputDataM_transformed,OutputDataM_transformed,val_split)
InputDataTrainL,InputDataValidL,OutputDataTrainL,OutputDataValidL = groupData(InputDataL_transformed,OutputDataL_transformed,val_split)

#for Tensorboard
#InputDataTrainS1000 = InputDataS_transformed[0:800]
#InputDataValidS1000 = InputDataS_transformed[800:1000]
#OutputDataTrainS1000 = OutputDataS_transformed[0:800]
#OutputDataValidS1000 = OutputDataS_transformed[800:1000]


#chooseModel
#modelName="kerasTunerModels/modelOne"
modelName="kerasTunerModels/modelTwo"

#loadModel
model = tf.keras.models.load_model("models/"+modelName)

#compileModel
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy'])


log_dir = "logs/fit/" + modelName + "08TrainValid" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=25)

#trainModel
def modelFit(epochs,bs,x,y,x_val,y_val):
    model.fit(x=x, 
          y=y, 
          epochs=epochs,
          batch_size = bs,
          validation_data=(x_val, y_val), 
          verbose = 0,
          callbacks=[es] #[tensorboard_callback] 
         )
    results = model.evaluate(x_val,y_val)
    print("Val loss, Val accuracy:", results)


#modelFit(200,1000,InputDataTrainS,OutputDataTrainS,InputDataValidS,OutputDataValidS)
#modelFit(200,1000,InputDataTrainM,OutputDataTrainM,InputDataValidM,OutputDataValidM)

modelFit(200,20000,InputDataTrainL,OutputDataTrainL,InputDataValidL,OutputDataValidL)

results = model.evaluate(InputData_transformed,OutputData_transformed)
print("Train loss, Train accuracy:", results)

#saveModel
#model.save("Models/"+modelName)