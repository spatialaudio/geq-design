"""
Function to get information of a model
    
    Parameters
    ----------
    modelName : string
        name of model  
    Returns
    -------
        
    Notes
    -----
    
"""

import tensorflow as tf

#chooseModel
modelName = "kerasTunerModels/modelOne"
#modelName="kerasTunerModels/modelTwo"

#loadModel
model = tf.keras.models.load_model("models/"+modelName)

#model summary
model.summary()

#model config
for i in range(len(model.layers)):
    print(model.layers[i].get_config())
