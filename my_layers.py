#! /usr/bin/python3

from tensorflow.keras.layers import Layer
from tensorflow import convert_to_tensor, expand_dims, concat
from numpy.linalg import norm
from my_datahanddlers import delta_encoding, delta_decoding

class pre_process_layer(Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)

    def build(self, inp_shape):
        pass

    def call(self, inputs):
        X = list()
        X_0 = list()
        N = list()
        for v in inputs:
            print(v)
            x,x_0 = delta_encoding(v.numpy())
            n = norm(x)
            X.append(convert_to_tensor(x/n))
            X_0.append(expand_dims(convert_to_tensor(x_0),1))
            N.append(expand_dims(convert_to_tensor(n),1))
        return [concat(X,axis=1), concat(X_0,axis=1), concat(N,axis=1)]

    def compute_output_shape(self, inp_shape):
        return inp_shape[:-1],inp_shape[-1] + 2

class pos_process_layer(pre_process_layer):
    def call(self, X, X_0, N):
        V = list()
        for x,x_0,n in zip(X,X_0,N):
            V.append(delta_decoding(x*n,x_0))
        return concat(V,axis=1)

    def compute_output_shape(self, inp_shape):
        return inp_shape[:-1],inp_shape[-1] - 2
