import numpy as np

def createInputData():
    
    data = np.random.uniform(-12,12,size=(42000,31))
    dataInt1 = ((2*np.random.randint(0,2,size=(3000,31))-1))*12
    dataInt2 = ((2*np.random.randint(0,2,size=(3000,31))-1))*12
    dataInt3 = np.random.randint(-1, 2, (1000, 31))*12
    dataInt4 = np.random.randint(-1, 2, (1000, 31))*12
    a1 = 12 * np.ones(31).reshape(31,1)
    a2 = -12 * np.ones(31).reshape(31,1)  
    a3 = np.append(np.tile([[12,-12]],15),12).reshape(31,1)
    a4 = a = np.append(np.tile([[-12,12]],15),-12).reshape(31,1)
    a5 = np.array([12,-12,-12,-12,12,12,12,-12,-12,-12,12,12,12,-12,-12,-12,12,12,12,-12,-12,-12,12,12,12,-12,-12,-12,12,12,12]).reshape(31,1)  
    a6 = np.array([-12,12,12,12,-12,-12,-12,12,12,12,-12,-12,-12,12,12,12,-12,-12,-12,12,12,12,-12,-12,-12,12,12,12,-12,-12,-12]).reshape(31,1)  
    a7 = a = np.append(np.tile([[12,12,-12]],10),12).reshape(31,1)
    a8 = np.append(np.tile([[-12,-12,12]],10),-12).reshape(31,1)
    a9 = np.append(np.tile([[12,-12,-12,-12]],7),[12,-12,-12]).reshape(31,1)
    a10 = np.append(np.tile([[-12,12,12,12]],7),[-12,12,12]).reshape(31,1)
    a = np.array([[a1],[a2],[a3],[a4],[a5],[a6],[a7],[a8],[a9],[a10]]).reshape(10,31)
    
    data1 = np.r_[a,dataInt1,dataInt3,data,dataInt2,dataInt4]
    np.random.shuffle(data1)

    data = np.r_[data1,a]
    
    return data

# define data
data = np.asarray(createInputData(), dtype=np.float64,)
# save to csv file
np.savetxt('dataInput.csv', data, delimiter=',')
