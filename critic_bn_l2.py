import tensorflow as tf
import numpy as np
from tensorflow import keras

class Critic(keras.Model):
    def __init__(self, units=32, activation='relu', weight_decay=1e-4, **kwargs):
        super().__init__(Critic, **kwargs)
        self.hidden1 = keras.layers.Dense(units, use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation(activation)
        self.hidden2= keras.layers.Dense(units, use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.Activation(activation)
        self.hidden3 = keras.layers.Dense(units, use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn3 = keras.layers.BatchNormalization()
        self.act3 = keras.layers.Activation(activation)
        self.output_layer = keras.layers.Dense(1)
    
    def call(self, inputs_, training=False):
        input_obs, input_action = inputs_
        hidden = self.hidden1(input_obs)
        hidden = self.bn1(hidden, training=training)
        hidden = self.act1(hidden)
        concat = keras.layers.concatenate([hidden, input_action])
        hidden = self.hidden2(concat)
        hidden = self.bn2(hidden, training=training)
        hidden = self.act2(hidden)
        hidden = self.hidden3(hidden)
        hidden = self.bn3(hidden, training=training)
        hidden = self.act3(hidden)
        out = self.output_layer(hidden)
        return out