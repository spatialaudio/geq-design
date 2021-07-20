"""
Function for creating a neural network

    Parameters
    ----------
        
    Returns
    -------

    returns keras sequential model

    Notes
    -----
    save the created model 
"""
import tensorflow as tf

def create_model() :
    return tf.keras.models.Sequential([
      tf.keras.layers.Dense(62,input_shape = (31,),activation='tanh'),
      tf.keras.layers.Dense(62,activation="tanh"),
      tf.keras.layers.Dense(31,activation="linear"),
    ])

modelName = "ModelSmall"

model = create_model()

model.summary()

model.save("Models/new/"+modelName)