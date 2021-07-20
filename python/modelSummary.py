import tensorflow as tf

#chooseModel
modelName = "kerasTunerModels/modelOne"
#modelName="ModelLarge"

#loadModel
model = tf.keras.models.load_model("models/"+modelName)

#model summary
model.summary()

#model config
for i in range(len(model.layers)):
    print(model.layers[i].get_config())
