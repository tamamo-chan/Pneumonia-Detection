from keras import backend as K
from keras.layers import Layer
from keras.initializers import Zeros, Ones
from keras import regularizers
from keras import activations

class myDense(Layer):
    def __init__(self, units, activation = None, use_bias=True, kernel_regularizer= None, **kwargs):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias  #implemented with zeroes
        self.kernel_regularizer = regularizers.get(kernel_regularizer) 
        super(myDense, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.kernel_regularizer != None:
            self.kernel = self.add_weight(name='kernel',
                                    shape=(input_shape[-1],self.units),
                                    initializer='uniform',
                                    regularizer=self.kernel_regularizer,    
                                    trainable=True)
        else:
            self.kernel = self.add_weight(name='kernel',
                                    shape=(input_shape[-1],self.units),
                                    initializer='uniform',
                                    trainable=True)


        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units, ),
                                        initializer=Zeros(),
                                        name='bias')
        else:
            self.bias = None

        super(myDense, self).build(input_shape) # Be sure to call this at the end

    def call(self, x):
        output = K.dot(x, self.kernel)

        if self.use_bias:
            output = K.bias_add(output , self.bias, data_format="channels_last")
        if self.activation == 'relu':
            output = activations.relu(output)
        elif self.activation == 'sigmoid':
            output = activations.sigmoid(output)
        
        return output

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


    def get_config(self):

        base_config = super(myDense, self).get_config()
        base_config['activation'] = self.activation
        base_config['units'] = self.units
        base_config['use_bias'] = self.use_bias
        base_config['kernel_regularizer'] = self.kernel_regularizer
        return base_config