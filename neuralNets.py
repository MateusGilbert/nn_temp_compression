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

def buildNN(layer_conf):#,out_sp):
	#in == input
	#cl == conv. layer
	#dl == dense layer
	#do == drop out
	#mp == max pool.
	#ap == av. pool.
	#fl == flatten
	#ls == lstm
	#rs == reshape
	#up == upsampling
	#ct == conv. transpose
	neural_net = []
	for tp, num, act in layer_conf:
		if tp == 'in':
			neural_net.append(Input(num))
		elif tp == 'cl':
			n_filters ,k_size, stride, padding = num
			if act == 'selu':
				k_init = 'lecun_normal'
			else:
				k_init = 'glorot_uniform'
			if padding:
				neural_net.append(
					Conv1D(filters=n_filters, kernel_size=k_size, strides=stride, activation=act,
						kernel_initializer=k_init,padding=padding)
				)
			else:
				neural_net.append(
					Conv1D(filters=n_filters, kernel_size=k_size, strides=stride, activation=act,
						kernel_initializer=k_init)
				)
		elif tp == 'dl':
			if act == 'selu':
				k_init = 'lecun_normal'
			else:
				k_init = 'glorot_uniform'
			neural_net.append(
				Dense(num, activation=act, kernel_initializer=k_init)
			)
		elif tp == 'ls':
			n_lstm_cells, r_seq_opt = num
			neural_net.append(
				LSTM(n_lstm_cells, return_sequences=r_seq_opt)
			)
		elif tp == 'do':
			neural_net.append(Dropout(num))
		elif tp == 'mp':
			neural_net.append(MaxPool1D(num))
		elif tp == 'ap':
			neural_net.append(AvPool1D(num))
		elif tp == 'bn':
			neural_net.append(BatchNorm())
		elif tp == 'fl':
			neural_net.append(Flatten())
		elif tp == 'rs':
			neural_net.append(Reshape(num))
		elif tp == 'up':
			neural_net.append(UpSamp1D(num))
		elif tp == 'ct':
			n_filters ,k_size, stride, padding = num
			if padding:
				neural_net.append(Conv1DT(n_filters, kernel_size=k_size, strides=stride, activation=act,
					padding=padding))
			else:
				neural_net.append(Conv1DT(n_filters, kernel_size=k_size, strides=stride, activation=act))
		else:
			print('Unkown Layer!!! {} was not added.'.format(tp))
	model = Sequential(neural_net)
	return model

if __name__ == '__main__':
	net1 = [('in',(100,1),None),('cl',(1,5,3,None),'selu'),('ap',3,None),('cl',(1,3,1,None),'selu'),('fl',None,None),('dl',10,'selu'),('dl',1,'softmax')]
	nn1 = buildNN(net1)
	nn1.summary()
	net2 = [('in',(30,),None),('dl',55,'selu'),('do',.1,None),('dl',75,'selu'),('do',.1,None),('dl',100,'sigmoid'),('rs',(100,1),None),('ls',(5,True),None),('ls',(1,True),None)]
	nn2 = buildNN(net2)
	nn2.summary()
