#! /usr/bin/python3

from tensorflow.keras.layers import Dense, Dropout, Conv1D,\
 BatchNormalization as BatchNorm, MaxPooling1D as MaxPool1D,\
 AveragePooling1D as AvPool1D, Flatten, Input, LSTM, Reshape,\
 UpSampling1D as UpSamp1D, Conv1DTranspose as Conv1DT
from tensorflow.keras import Sequential
import tensorflow.nn as nn_utils
from numpy import exp

#sigmoid implementation at /home/mateusgilbert/venv/lib/python3.6/site-packages/tensorflow/python/ops/math_ops.py
def custom_sigmoid(x, a=2.):
	if isinstance(x,str):
		return ' with constant: {}'.format(a)
	return nn_utils.sigmoid(a*x)

def get_input_shape(layers):
	layers.reverse()
	input_shape = layers.pop()
	layers.reverse()
	return (input_shape,)

from tensorflow.keras.layers import Dense, Dropout, Conv1D,\
 BatchNormalization as BatchNorm, MaxPooling1D as MaxPool1D,\
 AveragePooling1D as AvPool1D, Flatten, Input, LSTM, Reshape,\
 UpSampling1D as UpSamp1D, Conv1DTranspose as Conv1DT,\
 RepeatVector, TimeDistributed, Conv2D, MaxPooling2D as MaxPool2D,\
 AveragePooling2D as AvPool2D, Conv2DTranspose as Conv2DT,\
 UpSampling2D as UpSamp2D
from tensorflow.keras.models import Model
from keras.layers.merge import concatenate as ConcatL
import tensorflow.nn as nn_utils
from numpy import exp

def get_input_shape(layers):
	layers.reverse()
	input_shape = layers.pop()
	layers.reverse()
	return (input_shape,)

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
	neural_net = prev_layers
	for tp, num, act in layer_conf:
		if tp == 'in':
			if not neural_net:
				inputs = Input(num)
				neural_net = inputs
			else:
				return None
		elif tp == 'cv':
			n_filters ,k_size, stride, padding = num
			if act == 'selu':
				k_init = 'lecun_normal'
			else:
				k_init = 'glorot_uniform'
			if padding:
				neural_net = Conv1D(filters=n_filters, kernel_size=k_size, strides=stride, activation=act,
						kernel_initializer=k_init,padding=padding)(neural_net)
			else:
				neural_net = Conv1D(filters=n_filters, kernel_size=k_size, strides=stride, activation=act,
						kernel_initializer=k_init)(neural_net)
		elif tp == 'cv2d':
			n_filters, k_size, stride, padding = num
			if act == 'selu':
				k_init = 'lecun_normal'
			else:
				k_init = 'glorot_uniform'
			if isinstance(k_size, int):
				k_size = (k_size,k_size)
			if isinstance(stride,int):
				stride = (stride,stride)
			if padding:
				neural_net = Conv2D(filters=n_filters, kernel_size=k_size, strides=stride, activation=act,
														kernel_initializer=k_init,padding=padding)(neural_net)
			else:
				neural_net = Conv2D(filters=n_filters, kernel_size=k_size, strides=stride, activation=act,
														kernel_initializer=k_init)(neural_net)
		elif tp == 'dl':
			if act == 'selu':
				k_init = 'lecun_normal'
			else:
				k_init = 'glorot_uniform'
			neural_net = Dense(num, activation=act, kernel_initializer=k_init)(neural_net)
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
			n_filters ,k_size, stride, padding = num
			if act == 'selu':
				k_init = 'lecun_normal'
			else:
				k_init = 'glorot_uniform'
			if padding:
				neural_net.append = Conv1DT(n_filters, kernel_size=k_size, strides=stride, activation=act,
					padding=padding, kernel_initializer=k_init)(neural_net)
			else:
				neural_net = Conv1DT(n_filters, kernel_size=k_size, strides=stride, activation=act, kernel_initializer=k_init)(neural_net)
		elif tp == 'ct2d':
			n_filters, k_size, stride, padding = num
			if act == 'selu':
				k_init = 'lecun_normal'
			else:
				k_init = 'glorot_uniform'
			if isinstance(k_size, int):
				k_size = (k_size,k_size)
			if isinstance(stride,int):
				stride = (stride,stride)
			if padding:
				neural_net = Conv2DT(filters=n_filters, kernel_size=k_size, strides=stride, activation=act,
														kernel_initializer=k_init,padding=padding)(neural_net)
			else:
				neural_net = Conv2DT(filters=n_filters, kernel_size=k_size, strides=stride, activation=act,
														kernel_initializer=k_init)(neural_net)
		elif tp == 'rv':
			neural_net = RepeatVector(num)(neural_net)
		elif tp == 'td':
			neural_net = TimeDistributed(Dense(num))(neural_net)
		else:
			print('Unkown Layer!!! {} was not added.'.format(tp))
	if not return_model:
		return neural_net
	return Model(inputs=inputs, outputs=[neural_net])

def multOutputsNN(root_layers, branches, return_model=True):
	if not isinstance(branches[0], list):
		return buildNN2(root_layers + branches, return_model=False)
	inputs = None
	if isinstance(root_layers, list):
		inputs = buildNN2([root_layers[0]], return_model=False)
		r_layers = buildNN2(root_layers[1:], return_model=False, prev_layers=inputs)
	else:
		r_layers = root_layers
	outputs = list()
	for layers in branches:
		outputs.append(buildNN2(layers, return_model=False, prev_layers=r_layers))
	if not return_model:
		return outputs, inputs
	return Model(inputs=inputs, outputs=outputs)

def multInputsNN(roots, leaf_layers, return_model=True):
	if not isinstance(roots[0], list):
		return buildNN2(roots + leaf_layers, return_model=False)
	inputs = list()
	inp_layers = list()
	for i,layers in enumerate(roots):
		inputs.append(buildNN2([layers[0]], return_model=False))
		inp_layers.append(buildNN2(layers[1:], return_model=False, prev_layers=inputs[i]))
		node = ConcatL(inp_layers)
	neural_net = buildNN2(leaf_layers, prev_layers=node, return_model=False)
	if not return_model:
		return neural_net, inputs
	return Model(inputs=inputs, outputs=neural_net)

if __name__ == '__main__':
	net1 = [('in',(100,1),None),('cl',(1,5,3,None),'selu'),('ap',3,None),('cl',(1,3,1,None),'selu'),('fl',None,None),('dl',10,'selu'),('dl',1,'softmax')]
	nn1 = buildNN(net1)
	nn1.summary()
	net2 = [('in',(30,),None),('dl',55,'selu'),('do',.1,None),('dl',75,'selu'),('do',.1,None),('dl',100,'sigmoid'),('rs',(100,1),None),('ls',(5,True),None),('ls',(1,True),None)]
	nn2 = buildNN(net2)
	nn2.summary()
