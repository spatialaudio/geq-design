"""
search for the best neural network model

    Parameters
    ----------
    modelName: string
        model what is to be searched and saved 

    Returns
    -------
    saves the searched model as tensforflow sequential model
        
    Notes
    -----
"""

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

modelName ="modelNew"

InputData = np.loadtxt("Data/trainValid/dataInputSmall.csv",delimiter=",")
OutputData = np.loadtxt ("Data/trainValid/dataOutputSmall.csv", delimiter=",")

scaler = MinMaxScaler(feature_range=(0, 1))

InputData_transformed = scaler.fit_transform(InputData)
OutputData_transformed = scaler.fit_transform(OutputData)

InputData_train = InputData_transformed[0:8000]  
OutputData_train =  OutputData_transformed[0:8000] 
InputData_test = InputData_transformed[8000:10000]  
OutputData_test = OutputData_transformed[8000:10000]

def model_builder(hp):
  model = keras.Sequential()

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_unitsLayer1 = hp.Int('units', min_value=31, max_value=620, step=31)
  hp_activation1 = hp.Choice('dense_activation',values=['relu', 'tanh', 'sigmoid'],default='relu')
  model.add(keras.layers.Dense(input_shape=(31,),units=hp_unitsLayer1, activation=hp_activation1))
  hp_activation2 = hp.Choice('dense_activation',values=['relu', 'tanh', 'sigmoid',"linear"],default='relu') 
  hp_unitsLayer2 = hp.Int('units', min_value=31, max_value=620, step=31)
  model.add(keras.layers.Dense(units=hp_unitsLayer2, activation=hp_activation2))
  model.add(keras.layers.Dense(31,activation="linear"))
  
  # Tune the learning rate for the optimizer
  # Choose an optimal value from ,0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4,1e-5])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['accuracy'])

  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=200,
                     factor=3,
                     directory='my_dir_two',
                     project_name='modelTwo')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)

tuner.search(InputData_train, OutputData_train, epochs=200,batch_size=1000,validation_data =(InputData_test,OutputData_test) , callbacks=[stop_early],verbose=1)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(InputData_train,OutputData_train, epochs=200,batch_size=1000,validation_data =(InputData_test,OutputData_test), verbose=0)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(InputData_train, OutputData_train, epochs=best_epoch,batch_size=1000,validation_data =(InputData_test,OutputData_test),verbose=0)

eval_result = hypermodel.evaluate(InputData_test, OutputData_test)
print("[test loss, test accuracy]:", eval_result)

hypermodel.summary()

hypermodel.save("Models/kerasTunerModels/"+modelName)

