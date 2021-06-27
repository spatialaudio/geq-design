import tensorflow as tf

def create_model() :
    return tf.keras.models.Sequential([
      tf.keras.layers.Dense(62,input_shape = (31,),activation='relu'),
      tf.keras.layers.Dense(31,activation="linear"),
      tf.keras.layers.Dense(31,activation="linear"),
    ])

modelName = "newModel"

model = create_model()

model.summary()

model.save("Models/"+modelName)