def groupData(inputData,outputData,val_split):
    inputDataTrain = inputData[0:round(inputData.shape[0]*val_split)]
    inputDataValid  = inputData[round(inputData.shape[0]*val_split):inputData.shape[0]]
    outputDataTrain = outputData[0:round(outputData.shape[0]*val_split)]
    outputDataValid  = outputData[round(outputData.shape[0]*val_split):outputData.shape[0]]
    
    return inputDataTrain,inputDataValid,outputDataTrain,outputDataValid