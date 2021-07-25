"""
Function for grouping a input/output dataset into train and valid data

    Parameters
    ----------
    inputData : ndarray
        input data set
    outputData : ndarray
        corresponding output data set
    val_split : float
        select how large the validation data portion is

    Returns
    -------
    inputDataTrain : ndarray
        train input data set
    inputDataValid : ndarray
        valid input data set
    outputDataTrain : ndarray
        train output data set
    outputDataValid : ndarray
        valid output data set
    Notes   
    -----
    
"""

def groupData(inputData,outputData,val_split):
    inputDataTrain = inputData[0:round(inputData.shape[0]*val_split)]
    inputDataValid  = inputData[round(inputData.shape[0]*val_split):inputData.shape[0]]
    outputDataTrain = outputData[0:round(outputData.shape[0]*val_split)]
    outputDataValid  = outputData[round(outputData.shape[0]*val_split):outputData.shape[0]]
    
    return inputDataTrain,inputDataValid,outputDataTrain,outputDataValid