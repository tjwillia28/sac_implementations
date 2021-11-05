import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow import keras

@tf.function
def squashing_fxn(mu, action, log_prob):
    """Bound actions to a finite range."""
    log_prob -= tf.reduce_sum(2*(np.log(2)-action-tf.nn.softplus(-2*action)), axis=1)
    mu = tf.tanh(mu)
    action = tf.tanh(action)
    return mu, action, log_prob

class Actor(keras.Model):
    def __init__(self,
                 action_scale, # action_scale is determined by the max action that can be taken in the environment
                 # when we modify this for the trading problem action_scale==1
                 min_log_sigma=-20,
                 max_log_sigma=2,
                 num_actions=1,
                 units=32,
                 activation='relu',
                 weight_decay=1e-4,
                 **kwargs):
        super().__init__(Actor, **kwargs)
        self.hidden1=keras.layers.Dense(units, use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay), input_shape=[3,1])
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation(activation)
        self.hidden2=keras.layers.Dense(units, use_bias=False, kernel_regularizer=keras.regularizers.l2(weight_decay))
        self.bn2 = keras.layers.BatchNormalization()
        self.act2 = keras.layers.Activation(activation)
        self.mu=keras.layers.Dense(num_actions)
        self.log_sigma=keras.layers.Dense(num_actions)
        
        self.min_log_sigma=min_log_sigma
        self.max_log_sigma=max_log_sigma
        self.action_scale=action_scale
    
    def call(self, input_obs, training=False):
        hidden = self.hidden1(input_obs)
        hidden = self.bn1(hidden, training=training)
        hidden = self.act1(hidden)
        hidden = self.hidden2(hidden)
        hidden = self.bn2(hidden, training=training)
        hidden = self.act2(hidden)
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        
        # Clip log sigma
        log_sigma=tf.clip_by_value(log_sigma, self.min_log_sigma, self.max_log_sigma)
        sigma=tf.exp(log_sigma)
        
        # Create Gaussian Policy
        actor_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma**2)
        action = actor_dist.sample()
        log_prob = actor_dist.log_prob(action)
        
        # Restrict actions to bounded range
        mu, action, log_prob = squashing_fxn(mu, action, log_prob)
        
        # Scale mu, action
        mu *= self.action_scale
        action *= self.action_scale
        
        return mu, action, log_prob
    