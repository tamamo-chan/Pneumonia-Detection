from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import initializers
from keras.utils import conv_utils
from keras.engine.base_layer import InputSpec


class myConv2d(Layer):

    # Only have the most needed features
    def __init__(self, filters,
                 kernel_size,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(myConv2d, self).__init__(**kwargs)

        # Initialize the corresponding class variables
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2,'kernel_size')

        self.strides = conv_utils.normalize_tuple((1, 1), 2,'strides')

        self.data_format = "channels_last"
        self.dilation_rate = conv_utils.normalize_tuple((1, 1), 2,'dilation_rate')
        self.activation = activations.get('relu')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):

        # Only interested in channels_last format, so channel_axis is predetermined
        channel_axis = -1
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        # Create the weights and add bias
        self.kernel = self.add_weight(shape=kernel_shape,
                                      name='kernel',
                                      initializer=self.kernel_initializer)
        self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias')

        super(myConv2d, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        outputs = K.bias_add(
            outputs,
            self.bias,
            data_format=self.data_format)

        return self.activation(outputs)


    def compute_output_shape(self, input_shape):

        space = input_shape[1:-1]
        new_space = []

        for i in range(len(space)):
            new_filter_size = (self.kernel_size[i] - 1) * self.dilation_rate[i] + 1 # (3 - 1) * 1 + 1 = 3
            output_length = space[i] - new_filter_size + 1 # 224 - 3 + 1= 222
            new_space.append((output_length + self.strides[i] - 1) // self.strides[i]) # 222 + 0 // 1 = 220

        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def get_config(self):
        base_config = super(myConv2d, self).get_config()
        base_config['filters'] = self.filters
        base_config['kernel_size'] = self.kernel_size
        base_config['kernel_initializer'] = self.kernel_initializer
        base_config['bias_initializer'] = self.bias_initializer
        base_config['bias_initializer'] = self.bias_initializer

        return base_config