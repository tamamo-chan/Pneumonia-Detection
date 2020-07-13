from keras import backend as K
from keras.layers import *
from keras.layers import Layer
from keras.models import Model
import tensorflow as tf
import math

class maxpool(Layer):
    def __init__(self, pool_size=(2,2), strides=(2,2), padding='valid',
                 data_format="channels_last", **kwargs):
        super(maxpool, self).__init__(**kwargs)
        self.pool_size = tuple(pool_size)
        self.strides = tuple(strides)
        self.padding = padding
        self.data_format = data_format

    def call(self, x):
        output = K.pool2d(x, self.pool_size, self.strides,
                          self.padding, self.data_format,
                          pool_mode='max')
        return output

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], math.floor(input_shape[1]/2), math.floor(input_shape[2]/2), input_shape[3]]
        return tuple(output_shape)

    def get_config(self):
        base_config = super(maxpool, self).get_config()
        base_config['pool_size'] = self.pool_size
        base_config['strides'] = self.strides
        base_config['padding'] = self.padding
        base_config['data_format'] = self.data_format
        return base_config