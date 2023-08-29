#! /usr/bin/python3

from tensorflow.keras.layers import Dense, Dropout, Conv1D,\
 BatchNormalization as BatchNorm, MaxPooling1D as MaxPool1D,\
 AveragePooling1D as AvPool1D, Flatten, Input, LSTM, Reshape,\
 UpSampling1D as UpSamp1D, Conv1DTranspose as Conv1DT,\
 RepeatVector, TimeDistributed, Conv2D, MaxPooling2D as MaxPool2D,\
 AveragePooling2D as AvPool2D, Conv2DTranspose as Conv2DT,\
 UpSampling2D as UpSamp2D, Lambda, Layer
from my_layers import UpSmpConv, UpDenseConv, HybridConv
from tensorflow.keras.models import Model, load_model
#from keras.layers.merge import concatenate as ConcatL
from keras.layers import concatenate as ConcatL
from tensorflow.keras import regularizers as tf_regs
from tensorflow import expand_dims
import tensorflow.nn as nn_utils
from numpy import exp
import re
#from orth_reg import *


#sigmoid implementation at /home/mateusgilbert/venv/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py
def custom_sigmoid(x, a=1.01):
    if isinstance(x,str):
        return ' with constant: {}'.format(a)
    return a*nn_utils.sigmoid(x)

def get_input_shape(layers):
    layers.reverse()
    input_shape = layers.pop()
    layers.reverse()
    return (input_shape,)

#const models######################
def const_models(inp,brcs,outs,label):
    models = list()
    for branch in brcs:
        b_id, b_config = branch
        for output in outs:
            o_id, o_config = output
            models.append((label + b_id + o_id, [inp, [b_config, b_config, b_config, b_config], o_config]))
    return models

def add_lregulirizer(net, regs, layers):
    if not isinstance(regs,list):
        regs = [regs]
        layers = [layers]
    assert len(regs) == len(layers)
    for (label,r_op, k),l_ids in zip(regs,layers):
        if isinstance(l_ids, int):
            l_ids = [l_ids]
        if r_op == 'l1':
            reg = tf_regs.L1(l1=k)
        elif r_op == 'l2':
            reg = tf_regs.L2(l2=k)
        elif r_op == 'l1l2':
            l1,l2 = k
            reg = tf_regs.L1L2(l1=l1,l2=l2)
        elif r_op == 'rort':
            reg = OrthogonalRegularizer(factor=k, mode='rows')
        elif r_op == 'cort':
            reg = OrthogonalRegularizer(factor=k, mode='columns')
        else:
            reg = None
        for l_id in l_ids:
            setattr(net.layers[l_id], label, reg)
    return

class ShDenLayer(Layer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        if self.units < 0:
            raise ValueError(
                f"Received an invalid value for `units`, expected "
                f"a positive integer. Received: units={units}"
            )

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self,input_shape):
        #libraries needed for building layer
        import tensorflow as tf
        from keras import backend

        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-2])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer "
                "should be defined. Found None. "
                f"Full input shape received: {input_shape}"
            )
        #self.input_spec = tf.InputSpec(min_ndim=3, axes={-2: last_dim})

        self.W = self.add_weight(shape=[last_dim,self.units], 
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=True)
        self.b = self.add_weight(shape=[self.units,],
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  trainable=True) \
        if self.use_bias else None
        self.built = True

    def call(self, inputs):
        assert inputs.shape.rank == 3,'Wrong size! Dim must be equal to 3'
        assert inputs.shape[-2] == self.W.shape[0],f'Sizes don\'t match! {inputs.shape[-2]} inputs for {self.W.shape[0]} neurons'

        n = inputs.shape[-1]
        output = concat([expand_dims(matmul(inputs[:,:,i],self.W) + self.b, axis=2) for i in range(n)],2)
        
        if self.activation:
            return self.activation(output)
        return output

def buildNN(layer_conf, return_model=True, prev_layers=None):
    #in == input
    #cv == conv. layer
    #dl == dense layer
    #do == drop out
    #mp == max pool.
    #mp2d == max pool. 2D
    #ap == av. pool.
    #ap2d == av. pool. 2D
    #fl == flatten
    #ls == lstm
    #rs == reshape
    #up == upsampling
    #up2d == upsampling
    #ct == conv. transpose
    #rv == repeat vector
    #td == time distributed
    #cv2d == conv. 2D
    #ct2d == conv. transpose 2D
    #conc == concatenate outputs
    neural_net = prev_layers
    for tp, num, act in layer_conf:
        if tp == 'in':
            if not neural_net:
                inputs = Input(num)
                neural_net = inputs
            else:
                return None

        elif tp == 'cv':
            if len(num) == 4:
                nout_channels, k_size, stride, padding = num
                groups = 1
            else:
                nout_channels, k_size, stride, padding, groups = num
            if isinstance(padding,str):
                dilation=1
            else:
                padding,dilation = padding
            if act == 'selu':
                k_init = 'lecun_normal'
            else:
                k_init = 'glorot_uniform'
            if padding:
                neural_net = Conv1D(filters=nout_channels, kernel_size=k_size, strides=stride, activation=act,
                        kernel_initializer=k_init,padding=padding,dilation_rate=dilation,groups=groups)(neural_net)
            else:
                neural_net = Conv1D(filters=nout_channels, kernel_size=k_size, strides=stride, activation=act,
                        kernel_initializer=k_init,dilation_rate=dilation,groups=groups)(neural_net)

        elif tp == 'cv2d':
            nout_channels, k_size, stride, padding = num
            if act == 'selu':
                k_init = 'lecun_normal'
            else:
                k_init = 'glorot_uniform'
            if isinstance(k_size, int):
                k_size = (k_size,k_size)
            if isinstance(stride,int):
                stride = (stride,stride)
            if padding:
                neural_net = Conv2D(filters=nout_channels, kernel_size=k_size, strides=stride, activation=act,
                                                        kernel_initializer=k_init,padding=padding)(neural_net)
            else:
                neural_net = Conv2D(filters=nout_channels, kernel_size=k_size, strides=stride, activation=act,
                                                        kernel_initializer=k_init)(neural_net)

        elif tp == 'dl':
            if act == 'selu':
                k_init = 'lecun_normal'
            else:
                k_init = 'glorot_uniform'
            neural_net = Dense(num, activation=act, kernel_initializer=k_init)(neural_net)

        elif tp == 'dlsh':
            if act == 'selu':
                k_init = 'lecun_normal'
            else:
                k_init = 'glorot_uniform'
            neural_net = ShDenLayer(num, activation=act, kernel_initializer=k_init)(neural_net)

        elif tp == 'ls':
            n_lstm_cells, r_seq_opt = num
            neural_net = LSTM(n_lstm_cells, return_sequences=r_seq_opt)(neural_net)

        elif tp == 'do':
            neural_net = Dropout(num)(neural_net)

        elif tp == 'mp':
            neural_net = MaxPool1D(num)(neural_net)

        elif tp == 'mp2d':
            if isinstance(num, int):
                num = (num,num)
            neural_net = MaxPool2D(num)(neural_net)

        elif tp == 'ap':
            neural_net = AvPool1D(num)(neural_net)

        elif tp == 'ap2d':
            if isinstance(num, int):
                num = (num,num)
            neural_net = AvPool2D(num)(neural_net)

        elif tp == 'bn':
            neural_net = BatchNorm()(neural_net)

        elif tp == 'fl':
            neural_net = Flatten()(neural_net)

        elif tp == 'rs':
            neural_net = Reshape(num)(neural_net)

        elif tp == 'up':
            neural_net = UpSamp1D(num)(neural_net)

        elif tp == 'up2d':
            if isinstance(num, int):
                num = (num,num)
            neural_net = UpSamp2D(num)(neural_net)

        elif tp == 'ct':
            if len(num) > 4:
                nout_channels, k_size, stride, padding, group = num
            else:
                nout_channels, k_size, stride, padding = num
                group = False
            if isinstance(padding,str):
                dilation=1
            elif isinstance(padding,tuple) or isinstance(padding,list):
                padding,dilation = padding
            else:
                padding = 'valid'; dilation=1
            if act == 'selu':
                k_init = 'lecun_normal'
            else:
                k_init = 'glorot_uniform'
            nin_channels = neural_net.type_spec.shape[-1]
            if group and not (nin_channels % nout_channels):                                #check if we can split channels equaly
                #nin_channels = neural_net.type_spec.shape[-1]
                n = nin_channels // nout_channels
                if n > 1:
                    for i in range(n):
                        aux_layers = [Lambda(lambda x: x[:,:,i*nout_channels:(i+1)*nout_channels-1])
                                    for i in range(nout_channels)]
                else:
                    aux_layers = [Lambda(lambda x: expand_dims(x[:,:,i], axis=2)) for i in range(nout_channels)]
                inps = [layer(neural_net) for layer in aux_layers]
                outputs = [
                    Conv1DT(filters=1, kernel_size=k_size, strides=stride, activation=act,
                            kernel_initializer=k_init,padding=padding,dilation_rate=dilation)(i)
                            for i in inps
                    ]
                neural_net = ConcatL(outputs)
            elif group and (nin_channels == group):
                aux_layers = [Lambda(lambda x: expand_dims(x[:,:,i],axis=2)) for i in range(nin_channels)]
                inps = [layer(neural_net) for layer in aux_layers]
                outputs = [
                    Conv1DT(filters=nout_channels//group, kernel_size=k_size, strides=stride, activation=act,
                            kernel_initializer=k_init,padding=padding,dilation_rate=dilation)(i)
                            for i in inps
                    ]
                neural_net = ConcatL(outputs)
            else:
                neural_net = Conv1DT(nout_channels, kernel_size=k_size, strides=stride, activation=act,
                    padding=padding, kernel_initializer=k_init,dilation_rate=dilation)(neural_net)

        elif tp == 'ct2d':
            nout_channels, k_size, stride, padding = num
            if act == 'selu':
                k_init = 'lecun_normal'
            else:
                k_init = 'glorot_uniform'
            if isinstance(k_size, int):
                k_size = (k_size,k_size)
            if isinstance(stride,int):
                stride = (stride,stride)
            if padding:
                neural_net = Conv2DT(filters=nout_channels, kernel_size=k_size, strides=stride, activation=act,
                                                        kernel_initializer=k_init,padding=padding)(neural_net)
            else:
                neural_net = Conv2DT(filters=nout_channels, kernel_size=k_size, strides=stride, activation=act,
                                                        kernel_initializer=k_init)(neural_net)

        elif tp == 'rv':
            neural_net = RepeatVector(num)(neural_net)

        elif tp == 'td':
            neural_net = TimeDistributed(Dense(num))(neural_net)

        elif tp == 'conc':
            neural_net = ConcatL(neural_net, axis=num)

        elif tp == 'h_cv':
            if len(num) > 4:
                dilations, k_size, stride, padding, group = num
            else:
                dilations, k_size, stride, padding = num
                group = False
            if act == 'selu':
                k_init = 'lecun_normal'
            else:
                k_init = 'glorot_uniform'
            if group and not (neural_net.type_spec.shape[-1] % len(dilations)):
                n_ch = neural_net.type_spec.shape[-1]
                n_dil = len(dilations)
                n = n_ch // n_dil
                if n > 1:
                    for i in range(n):
                        aux_layers = [Lambda(lambda x: x[:,:,i*n_dil:(i+1)*n_dil-1]) for i in range(n_dil)]
                else:
                    aux_layers = [Lambda(lambda x: expand_dims(x[:,:,i], axis=2)) for i in range(n_dil)]
                inps = [layer(neural_net) for layer in aux_layers]
                outputs = [
                    Conv1D(filters=1, kernel_size=k_size, strides=stride, activation=act,                    #para deixar mais complexo, variar esse filters
                            kernel_initializer=k_init,padding=padding,dilation_rate=dil)(i)
                            for dil,i in zip(dilations, inps)
                    ]
            else:
                outputs = [
                    Conv1D(filters=1, kernel_size=k_size, strides=stride, activation=act,
                            kernel_initializer=k_init,padding=padding,dilation_rate=dil)(neural_net)
                            for dil in dilations
                    ]
            neural_net = ConcatL(outputs)

        elif tp == 'up_conv':
            if len(num) > 3:
                nout_channels, k_size, stride, depth_mult = num
            else:
                nout_channels, k_size, stride = num
                depth_mult = 1
            neural_net = UpSmpConv(nout_channels, k_size, stride, act, depth_mult)(neural_net)

        elif tp == 'd_conv':
            if len(num) > 4:
                if len(num) == 5:
                    inp_shape, nout_channels, k_size, stride, depth_mult = num
                    d_out = None
                else:
                    inp_shape, nout_channels, k_size, stride, depth_mult, d_out = num
            else:
                inp_shape, nout_channels, k_size, stride = num
                depth_mult = 1
                d_out = None
            neural_net = UpDenseConv(inp_shape, nout_channels, k_size, stride, act, depth_mult, d_out)(neural_net)

        elif tp == 'h_conv2':
            if len(num) > 3:
                dil_config, nout_channels, k_size, depth_mults = num
            else:
                dil_config, nout_channels, k_size = num
                depth_mults = []
            neural_net = HybridConv(dil_config, nout_channels, k_size, act, depth_mults)(neural_net)

        else:
            print('Unkown Layer!!! {} was not added.'.format(tp))

    if not return_model:
        return neural_net
    return Model(inputs=inputs, outputs=[neural_net])

def multOutputsNN(root_layers, branches, return_model=True):
    if not isinstance(branches[0], list):
        return buildNN(root_layers + branches, return_model=False)
    inputs = None
    if isinstance(root_layers, list):
        inputs = buildNN([root_layers[0]], return_model=False)
        r_layers = buildNN(root_layers[1:], return_model=False, prev_layers=inputs)
    else:
        r_layers = root_layers
    outputs = list()
    for layers in branches:
        outputs.append(buildNN(layers, return_model=False, prev_layers=r_layers))
    if not return_model:
        return outputs, inputs
    return Model(inputs=inputs, outputs=outputs)

def multInputsNN(roots, leaf_layers, return_model=True):
    if not isinstance(roots[0], list):
        return buildNN(roots + leaf_layers, return_model=False)
    inputs = list()
    inp_layers = list()
    for i,layers in enumerate(roots):
        inputs.append(buildNN([layers[0]], return_model=False))
        inp_layers.append(buildNN(layers[1:], return_model=False, prev_layers=inputs[i]))
    node = ConcatL(inp_layers)
    neural_net = buildNN(leaf_layers, prev_layers=node, return_model=False)
    if not return_model:
        return neural_net, inputs
    return Model(inputs=inputs, outputs=neural_net)

#specific for my implementation
def praae_init(praae, ae_path):
    base = load_model(ae_path)
    W = base.get_weights()
    W_len = len(W)
    j = 0
    for i in range(W_len // 2):
        while not re.search('^dense',praae.layers[i + j].name):            #######ajeitar, soh estah compativel c/ tot con AAE
            j += 1
        praae.layers[i + j].set_weights([W[2*i],W[2*i + 1]])
    del base, W
    return praae

#get pretrained layers:
def get_pretrained(dest_net,net_path,ex_layers,copy_to=None):
    assert isinstance(ex_layers,list) == True,'Bad argument!!! Don\'t know which layer to copy'
    base = load_model(net_path)

    layers = list(map(lambda i: base.layers[i], ex_layers))
    if (not isinstance(copy_to,list)) or (len(ex_layers) > len(copy_to)):
        copy_to = ex_layers

    dest_net.pr_layers = list()
    for from_layer, to_layer in zip(ex_layers, copy_to):
        W,b = from_layer.weights
        dest_net.layers[to_layer].set_weights([W,b])
        dest_net.pr_layers.append(dest_net.layers[to_layers])
    del W,b,base
    dest_net.new_layers = [layer for layer in dest_net.layers if layer not in dest_net.pr_layers]

    return dest_net

#add function that predefines weights

if __name__ == '__main__':
    netlist =  [('in', (100,), None), ('dl', 25, 'selu'), ('dl', 40, 'selu'), 
            ('dl', 50, 'selu'), ('rs', (50,1), None), ('h_cv', ([1,2,3,4], 3, 1, 'same') , 'selu'), ('ct', (16, 3, 2, 'same', 4), 'selu'), ('cv', (1,1,1,'same'), 'sigmoid'), ('rs', (100,), None)]
    net = buildNN(netlist)
    reg = [('kernel_regularizer', 'rort', .1), ('activity_regularizer', 'l1', .1)]
    layers = [1, 1]
    add_lregulirizer(net, reg, layers)
    net.summary()
    print(net.layers[1].kernel_regularizer)
    print(net.layers[1].weights)
    print(net.layers[14])
    print(net.layers[17])
#    ae_path = 'mod2test/AAE_1/bestAE.h5'
#    net1 = [('in',(100,1),None),('cl',(1,5,3,None),'selu'),('ap',3,None),('cl',(1,3,1,None),'selu'),('fl',None,None),('dl',10,'selu'),('dl',1,'softmax')]
#    nn1 = buildNN(net1)
#    nn1.summary()
#    net2 = [('in',(30,),None),('dl',55,'selu'),('do',.1,None),('dl',75,'selu'),('do',.1,None),('dl',100,'sigmoid'),('rs',(100,1),None),('ls',(5,True),None),('ls',(1,True),None)]
#    nn2 = buildNN(net2)
#    nn2.summary()
