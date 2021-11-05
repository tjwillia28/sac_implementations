import tensorflow as tf
import numpy as np
from tensorflow import keras

class Critic(keras.Model):
    def __init__(self, units=32, activation='relu', **kwargs):
        super().__init__(Critic, **kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.hidden3 = keras.layers.Dense(units, activation=activation)
        self.output_layer = keras.layers.Dense(1)
    
    def call(self, inputs_):
        input_obs, input_action = inputs_
        # action must be this form: action = np.array([0.0048969])
        # obs must be this form: obs [-0.43901418  0.89848014  0.77775782] type(obs) = np.array
        # to call predict method: critic.predict((obs[np.newaxis], action[np.newaxis]))
        hidden1 = self.hidden1(input_obs)
        concat = keras.layers.concatenate([hidden1, input_action])
        hidden2 = self.hidden2(concat)
        hidden3 = self.hidden3(hidden2)
        out = self.output_layer(hidden3)
        return out