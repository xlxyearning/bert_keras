# -*- coding: utf-8 -*-

import keras.backend as K
from keras.layers import Dropout, Embedding, Input
from keras.initializers import Ones, Zeros

class LayerNormalization(keras.layers.Layer):
    """Layer normalization layer.

    Attributes
        gamma: tensor by which to scale the input.
        beta: tensor with which to center the input.
        epsilon: small float added to variance to avoid dividing by zero.
    """

    def __init__(self, epsilon: float=1e-5, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='normalize_scale', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='normalize_bias', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        var = K.mean(K.square(x - mean), axis=-1, keepdims=True)
        res = (x - mean) / K.sqrt(var + self.epsilon)
        return self.gamma * res + self.beta
