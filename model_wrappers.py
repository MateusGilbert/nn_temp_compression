from tensorflow.keras.layers import Lambda, Input
from keras.layers import concatenate as ConcatL
from tensorflow.keras import Model
from tensorflow import concat
from my_layers import *

def my_wp(input_shape, sub_model):
    inp = Input(input_shape)
    enc_layer = pre_process_layer()(inp)
    model = sub_model(enc_layer[0])
    dec_layer = pos_process_layer()(model, *enc_layer[1:])
    return Model(inputs=inp, outputs=dec_layer)
