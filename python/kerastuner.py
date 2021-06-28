import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# test 2

InputData = np.loadtxt("Data/dataInputLarge2.csv",delimiter=",")
OutputData = np.loadtxt ("Data/dataOutputLarge2.csv", delimiter=",")

scaler = MinMaxScaler(feature_range=(0, 1))

InputData_transformed = scaler.fit_transform(InputData)
OutputData_transformed = scaler.fit_transform(OutputData)

InputData_train = InputData_transformed[0:18000]  
OutputData_train =  OutputData_transformed[0:18000] 
InputData_test = InputData_transformed[18000:22012]  
OutputData_test = OutputData_transformed[18000:22012]

def model_builder(hp):
  model = keras.Sequential()

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=31, max_value=6200, step=31)
  hp_activation = hp.Choice('dense_activation',values=['relu', 'tanh', 'sigmoid'],default='relu')
  model.add(keras.layers.Dense(input_shape=(31,),units=hp_units, activation=hp_activation))
  model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))
  model.add(keras.layers.Dense(31,activation=hp_activation))
  
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.1,0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-1,1e-2, 1e-3, 1e-4,1e-5])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['accuracy'])

  return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=200,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner.search(InputData_train, OutputData_train, epochs=200, validation_data =(InputData_test,OutputData_test) , callbacks=[stop_early],verbose=0)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(InputData_train,OutputData_train, epochs=200, validation_data =(InputData_test,OutputData_test), verbose=0)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(InputData_train, OutputData_train, epochs=best_epoch, validation_data =(InputData_test,OutputData_test),verbose=0)

eval_result = hypermodel.evaluate(InputData_test, OutputData_test)
print("[test loss, test accuracy]:", eval_result)
