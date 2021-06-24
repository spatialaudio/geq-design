import numpy as np
import tensorflow as tf
import datetime
import time
from sklearn.preprocessing import MinMaxScaler


#GetData
InputData = np.loadtxt("Data/dataInputLarge.csv",delimiter=",")
OutputData = np.loadtxt ("Data/dataOutputLarge.csv", delimiter=",")
InputDataSmall = np.loadtxt("Data/dataInput2.csv",delimiter=",")
OutputDataSmall = np.loadtxt ("Data/dataOutput2.csv", delimiter=",")
InputDataLarge = np.loadtxt("Data/dataInputLarge2.csv",delimiter=",")
OutputDataLarge = np.loadtxt ("Data/dataOutputLarge2.csv", delimiter=",")
InputDataBig = np.loadtxt("Data/dataInput.csv",delimiter=",")
OutputDataBig = np.loadtxt ("Data/dataOutput.csv", delimiter=",")


#TransformData
scaler = MinMaxScaler(feature_range=(0, 1))

InputData_transformed = scaler.fit_transform(InputData)
OutputData_transformed = scaler.fit_transform(OutputData)
InputDataSmall_transformed = scaler.fit_transform(InputDataSmall)
OutputDataSmall_transformed = scaler.fit_transform(OutputDataSmall)
InputDataLarge_transformed = scaler.fit_transform(InputDataLarge)
OutputDataLarge_transformed = scaler.fit_transform(OutputDataLarge)
InputDataBig_transformed = scaler.fit_transform(InputDataBig)
OutputDataBig_transformed = scaler.fit_transform(OutputDataBig)

OutputDataBig_retransformed = scaler.inverse_transform(OutputDataBig_transformed)         


#GroupData
InputDataTrain = InputData_transformed[0:9000]  
InputDataTest = InputData_transformed[9000:11006]  
OutputDataTrain = OutputData_transformed[0:9000] 
OutputDataTest = OutputData_transformed[9000:11006]

InputDataTrainS = InputDataSmall_transformed[0:900]  
InputDataTestS = InputDataSmall_transformed[900:1106]  
OutputDataTrainS = OutputDataSmall_transformed[0:900] 
OutputDataTestS = OutputDataSmall_transformed[900:1106]

InputDataTrainL = InputDataLarge_transformed[0:18000]  
InputDataTestL = InputDataLarge_transformed[18000:22012]  
OutputDataTrainL = OutputDataLarge_transformed[0:18000] 
OutputDataTestL = OutputDataLarge_transformed[18000:22012]

InputDataTrainBig = InputDataBig_transformed[0:40000] 
InputDataTestBig = InputDataBig_transformed[40000:50020] 
OutputDataTrainBig = OutputDataBig_transformed[0:40000] 
OutputDataTestBig = OutputDataBig_transformed[40000:50020] 


#chooseModel
#modelName = "Model"
modelName="ModelLarge"

#loadModel
model = tf.keras.models.load_model("Models/"+modelName)

#compileModel
model.compile(optimizer='adam',
              #loss=tf.keras.losses.MeanSquaredError(),
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#trainModel
def modelFit(epochs,batchSize,x,y,x_val,y_val):
    model.fit(x=x, 
          y=y, 
          epochs=epochs,
          batch_size = batchSize,
          validation_data=(x_val, y_val), 
          verbose = 0,
          #callbacks=[tensorboard_callback]
         )
    results = model.evaluate(InputDataTestBig,OutputDataTestBig)
    print("Test loss, Test accuracy:", results)


#modelFit(1000,2500,InputDataTrainBig,OutputDataTrainBig,InputDataTestBig,OutputDataTestBig)
#modelFit(1000,1000,InputDataTrainBig,OutputDataTrainBig,InputDataTestBig,OutputDataTestBig)
#modelFit(1000,500,InputDataTrainBig,OutputDataTrainBig,InputDataTestBig,OutputDataTestBig)
#modelFit(1000,200,InputDataTrainBig,OutputDataTrainBig,InputDataTestBig,OutputDataTestBig)
#modelFit(1500,200,InputDataTrain,OutputDataTrain,InputDataTest,OutputDataTest)
#modelFit(2500,100,InputDataTrainS,OutputDataTrainS,InputDataTestS,OutputDataTestS)
#modelFit(2000,200,InputDataTrainL,OutputDataTrainL,InputDataTestL,OutputDataTestL)

#saveModel
#model.save("Models/"+modelName)

print(InputDataBig[50010:50011])

print(scaler.inverse_transform(model(InputDataTestBig[10010:10011])))

print(OutputDataBig[50010:50011])

print(OutputDataBig[50010:50011]-scaler.inverse_transform(model(InputDataTestBig[10010:10011])))