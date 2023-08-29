#! /usr/bin/python3

import tensorflow.keras.layers as tl_layers #import Layer, Conv1D, Conv2D, UpSampling1D, UpSampling2D
from tensorflow import convert_to_tensor, expand_dims, concat
from numpy.linalg import norm
from my_datahanddlers import delta_encoding, delta_decoding
from keras.utils import tf_utils
from keras import Sequential
#
#class pre_process_layer(tl_layers.Layer):
#    def __init__(self, name=None, **kwargs):
#        super().__init__(**kwargs)
#
#    def build(self, inp_shape):
#        pass
#
#    def call(self, inputs):
#        X = list()
#        X_0 = list()
#        N = list()
#        for v in inputs:
#            print(v)
#            x,x_0 = delta_encoding(v.numpy())
#            n = norm(x)
#            X.append(convert_to_tensor(x/n))
#            X_0.append(expand_dims(convert_to_tensor(x_0),1))
#            N.append(expand_dims(convert_to_tensor(n),1))
#        return [concat(X,axis=1), concat(X_0,axis=1), concat(N,axis=1)]
#
#    @tf_utils.shape_type_conversion
#    def compute_output_shape(self, inp_shape):
#        return inp_shape[:-1],inp_shape[-1] + 2
#
#class pos_process_layer(pre_process_layer):
#    def call(self, X, X_0, N):
#        V = list()
#        for x,x_0,n in zip(X,X_0,N):
#            V.append(delta_decoding(x*n,x_0))
#        return concat(V,axis=1)
#
#    @tf_utils.shape_type_conversion
#    def compute_output_shape(self, inp_shape):
#        return inp_shape[:-1],inp_shape[-1] - 2

# conv layers

class UpSmpConv(tl_layers.Layer):
    def __init__(self, nout_channels, k_size, stride, activation, d_mult=1, conv_type=1, **kwargs):
        super(UpSmpConv,self).__init__(**kwargs)
        if activation == 'selu':
            k_init = 'lecun_normal'
        else:
            k_init = 'glorot_uniform'

        if conv_type == 1:
            self.conv_layer = tl_layers.SeparableConv1D(
                                        filters=nout_channels,
                                        kernel_size=k_size,
                                        strides=1,
                                        activation=activation,
                                        depthwise_initializer=k_init,
                                        pointwise_initializer=k_init,
                                        depth_multiplier=d_mult,
                                        padding='same',
                                    )
            self.up = tl_layers.UpSampling1D(
                                    stride
                                  )
        elif conv_type == 2:
            self.conv_layer = tl_layers.DepthwiseConv2D(
                                        filters=nout_channels,
                                        kernel_size=k_size,
                                        strides=(1,1),
                                        activation=activation,
                                        pointwise_initializer=k_init,
                                        depthwise_initializer=k_init,
                                        depth_multiplier=d_mult,
                                        padding='same',
                                    )
            self.up = tl_layers.UpSampling2D(
                                    stride, interpolation='bilinear'
                                  )
        else:
            raise 'This layer was designed for 1D and 2D convolutions.'
        self.nout_channels = nout_channels
        self.stride = stride

    def build(self, input_shape):
        super(UpSmpConv, self).build(input_shape)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        _, x, _ = input_shape
        return (input_shape[0], x*self.stride, self.nout_channels)

    def call(self, inputs):
        return self.conv_layer(self.up(inputs))

class UpDenseConv(tl_layers.Layer):
    def __init__(self, inp_shape, nout_channels, k_size, stride, activation, d_mult=1, dense_out=None, **kwargs):
        super(UpDenseConv,self).__init__(**kwargs)
        if activation == 'selu':
            k_init = 'lecun_normal'
        else:
            k_init = 'glorot_uniform'

        self.conv = tl_layers.SeparableConv1D(
                                    filters=nout_channels,
                                    kernel_size=k_size,
                                    strides=1,
                                    activation=activation,
                                    depthwise_initializer=k_init,
                                    pointwise_initializer=k_init,
                                    depth_multiplier=d_mult,
                                    padding='same',
                                )
        self.reshape = tl_layers.Reshape((inp_shape[0],))

        if isinstance(dense_out, (list,tuple)):
            self.up = list()
            for _ in range(inp_shape[1]):
                aux_up = [
                            tl_layers.Dense(
                                inp_shape[0]*mult_factor,
                                activation=activation,
                                kernel_initializer=k_init
                                )
                            for mult_factor in dense_out
                        ]
                self.up.append(
                       Sequential(aux_up + [ 
                            tl_layers.Dense(
                                    inp_shape[0]*stride,
                                    activation=activation,
                                    kernel_initializer=k_init
                                )])
                        )
        else:
            self.up = [
                        tl_layers.Dense(
                            dense_out if dense_out else inp_shape[0]*stride,        #gambiarra
                            activation=activation,
                            kernel_initializer=k_init
                            )
                        for _ in range(inp_shape[1])
                    ]
        self.nout_channels = nout_channels
        self.nin_channels = inp_shape[1]
        self.stride = stride

    def build(self, input_shape):
        super(UpDenseConv, self).build(input_shape)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        _, x, _ = input_shape
        return (input_shape[0], x*self.stride, self.nout_channels)

    def call(self, inputs):
        if self.nin_channels == 1:
            return self.conv(expand_dims(self.up[0](self.reshape(inputs)), axis=2))

        outputs = [expand_dims(self.up[i](inputs[:,:,i]), axis=2) for i in range(self.nin_channels)]
        outputs = concat(outputs, axis=2)
        return self.conv(outputs)

#here nout_channels is the actual number of output channels
class HybridConv(tl_layers.Layer):
    def __init__(self, dil_config, nout_channels, kernel_size=3, activation='selu', d_mults=[], **kwargs):
        super(HybridConv,self).__init__(**kwargs)
        assert len(dil_config) == len(nout_channels)
        if len(d_mults):
            assert len(d_mults) == len(nout_channels)

        if activation == 'selu':
            k_init = 'lecun_normal'
        else:
            k_init = 'glorot_uniform'

        self.convs = list()
        for i in range(len(dil_config)):
            if d_mults:
                self.convs.append(
                            tl_layers.SeparableConv1D(
                                    filters=nout_channels[i],
                                    activation=activation,
                                    kernel_size=kernel_size,
                                    dilation_rate=dil_config[i],
                                    depthwise_initializer=k_init,
                                    pointwise_initializer=k_init,
                                    depth_multiplier=d_mults[i],
                                    padding='same',
                                )
                        )
            else:
                self.convs.append(
                            tl_layers.SeparableConv1D(
                                    filters=nout_channels[i],
                                    activation=activation,
                                    kernel_size=k_size,
                                    depthwise_initializer=k_init,
                                    pointwise_initializer=k_init,
                                    dilation_rate=dil_config[i],
                                    padding='same',
                                )
                        )
        self.n = len(self.convs)
        self.nout_channels = nout_channels

    def build(self, input_shape):
        super(HybridConv, self).build(input_shape)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.nout_channels)

    def call(self, inputs):
        outputs = [self.convs[i](inputs) for i in range(self.n)]
        return concat(outputs, axis=2)

MY_LAYERS = {
            'UpSmpConv': UpSmpConv,
            'UpDenseConv': UpDenseConv,
            'HybridConv': HybridConv,
        }
