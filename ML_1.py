import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Simple linear regression with synthetic data
def build_model(my_learning_rate):
    model = tf.keras.models.sequential()
    model.add(tf.keras.layers.Dense(unit = 1, input_shape = (1,)))
    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = my_learning_rate),loss = "mean_squared_error",matrics = [tf.keras.matrics.RootMeanSquaredError()])
    return model


def train_model(model,feature,label,epochs,batch_size):
    history = model.fit(x=feature,y=label,batch_size=batch_size,epochs=epochs)
    trained_weight =  model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    epochs=history.epoch

    hist=pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return trained_weight, trained_bias, epochs, rmse
print("Defined create_model and train_model")
