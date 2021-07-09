import numpy as np
import tensorflow as tf
import datetime
import time
from sklearn.preprocessing import MinMaxScaler


#GetData
#only for test
InputDataTest = np.loadtxt("data/test/dataInputTest.csv",delimiter=",")
OutputDataTest = np.loadtxt ("data/test/dataOutputTest.csv", delimiter=",")
#for training
InputDataSmall = np.loadtxt("data/trainValid/dataInputSmall.csv",delimiter=",")
OutputDataSmall = np.loadtxt ("data/trainValid/dataOutputSmall.csv", delimiter=",")
InputDataMid = np.loadtxt("data/trainValid/dataInputMid.csv",delimiter=",")
OutputDataMid = np.loadtxt ("data/trainValid/dataOutputMid.csv", delimiter=",")
InputDataLarge = np.loadtxt("data/trainValid/dataInput.csv",delimiter=",")
OutputDataLarge = np.loadtxt ("data/trainValid/dataOutput.csv", delimiter=",")


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
def groupData(inputData,outputData,val_split):
    inputDataTrain = inputData[0:round(inputData.shape[0]*val_split)]
    inputDataValid  = inputData[round(inputData.shape[0]*val_split):inputData.shape[0]]
    outputDataTrain = outputData[0:round(outputData.shape[0]*val_split)]
    outputDataValid  = outputData[round(outputData.shape[0]*val_split):outputData.shape[0]]
    
    return inputDataTrain,inputDataValid,outputDataTrain,outputDataValid

val_split = 0.8
#InputDataTrainS,InputDataValidS,OutputDataTrainS,OutputDataValidS = groupData(InputDataS_transformed,OutputDataS_transformed,val_split)
#InputDataTrainM,InputDataValidM,OutputDataTrainM,OutputDataValidM = groupData(InputDataM_transformed,OutputDataM_transformed,val_split)
InputDataTrainL,InputDataValidL,OutputDataTrainL,OutputDataValidL = groupData(InputDataL_transformed,OutputDataL_transformed,val_split)

#for Tensorboard
InputDataTrainS = InputDataS_transformed[0:800]
InputDataValidS = InputDataS_transformed[800:1000]
OutputDataTrainS = OutputDataS_transformed[0:800]
OutputDataValidS = OutputDataS_transformed[800:1000]


#chooseModel

#modelName = "model"
#modelName="modelLarge"
#modelName="new/ModelSmall"
#modelName="new/ModelLarge"
#modelName="modelOne"
modelName="modelTwo"

#loadModel
model = tf.keras.models.load_model("models/kerasTunerModels/"+modelName)

#compileModel
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['accuracy'])


log_dir = "logs/fit/" + modelName + "EP50BS100DS1000" #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=50)

#trainModel
def modelFit(epochs,bs,x,y,x_val,y_val):
    model.fit(x=x, 
          y=y, 
          epochs=epochs,
          batch_size = bs,
          validation_data=(x_val, y_val), 
          verbose = 1,
        #  callbacks=[tensorboard_callback] #,es,tensorboard_callback]
         )
    results = model.evaluate(x_val,y_val)
    print("Test loss, Test accuracy:", results)

modelFit(50,100,InputDataTrainS,OutputDataTrainS,InputDataValidS,OutputDataValidS)

#modelFit(50,5000,InputDataTrainM,OutputDataTrainM,InputDataValidM,OutputDataValidM)

#modelFit(200,1000,InputDataTrainS,OutputDataTrainS,InputDataTestS,OutputDataTestS)
#modelFit(500,2000,InputDataTrainS,OutputDataTrainS,InputDataTestS,OutputDataTestS)
#modelFit(200,1000,InputDataTrainM,OutputDataTrainM,InputDataTestM,OutputDataTestM)
#modelFit(200,2000,InputDataTrainM,OutputDataTrainM,InputDataValidM,OutputDataValidM)
#modelFit(200,2000,InputDataTrainL,OutputDataTrainL,InputDataTestL,OutputDataTestL)
#modelFit(500,5000,InputDataTrainL,OutputDataTrainL,InputDataTestL,OutputDataTestL)

#saveModel
#model.save("Models/"+modelName)