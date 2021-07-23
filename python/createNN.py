"""
Function for creating a neural network

    Parameters
    ----------
    modelName : string
        name of the created model   
    Returns
    -------

    returns keras sequential model

    Notes
    -----
    saves the created model with it's name
"""
import tensorflow as tf

modelName = "ModelSmall"

def create_model() :
    return tf.keras.models.Sequential([
      tf.keras.layers.Dense(62,input_shape = (31,),activation='tanh'),
      tf.keras.layers.Dense(62,activation="tanh"),
      tf.keras.layers.Dense(31,activation="linear"),
    ])


model = create_model()

model.summary()

model.save("Models/new/"+modelName)