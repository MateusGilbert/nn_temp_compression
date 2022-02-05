#! /usr/bin/python3

from numpy import newaxis as nax
from numpy.random import normal, choice
from tensorflow import concat
from numpy import flip
import tensorflow as tf

#sigma is a hyperparameter

def jittering(dataset, sigma=.5, prob=None, addProb=1.):
	inputs = []
	targets = []
	num_batches = 0
	for x,y in dataset:
		num_batches += 1
		og_x = x.numpy()
		og_y = y.numpy()
		inputs.append(og_x)
		targets.append(og_y)
		#artificial time series signal
		if abs(addProb) > 1. or addProb < 0.:
			addProb = 1.
		if choice([True, False], p=[addProb,1.-addProb]):
			num_batches += 1
			if isinstance(sigma,list):
				if prob == None:
					prob = [1/len(sigma) for i in range(len(sigma))]
				inputs.append(og_x + normal(0,choice(sigma,1,p=prob).squeeze(),
					og_x.shape[0]))
			else:
				inputs.append(og_x + normal(0,sigma,og_x.shape[0]))
			targets.append(og_y)
	inputs = tf.data.Dataset.from_tensor_slices(inputs)
	targets = tf.data.Dataset.from_tensor_slices(targets)
	dts = tf.data.Dataset.zip((inputs,targets)).shuffle(buffer_size=num_batches)
	return dts, num_batches

#talvez reduzir o número de flips
def flipping(dataset,flipTg=True,addProb=1.):
	inputs = []
	targets = []
	num_batches = 0
	for x,y in dataset:
		num_batches += 1
		og_x = x.numpy()
		og_y = y.numpy()
		inputs.append(og_x)
		targets.append(og_y)
		#artificial time series signal
		if abs(addProb) > 1. or addProb < 0.:
			addProb = 1.
		if choice([True, False], p=[addProb,1.-addProb]):
			num_batches += 1
			inputs.append(flip(og_x))
			if flipTg:
				targets.append(flip(og_y))
			else:
				targets.append(og_y)
	inputs = tf.data.Dataset.from_tensor_slices(inputs)
	targets = tf.data.Dataset.from_tensor_slices(targets)
	dts = tf.data.Dataset.zip((inputs,targets)).shuffle(buffer_size=num_batches)
	return dts, num_batches

def scaling(dataset, sigma=.2,scale_y=False,addProb=1.):
	inputs = []
	targets = []
	num_batches = 0
	for x,y in dataset:
		num_batches += 1
		og_x = x.numpy()
		og_y = y.numpy()
		inputs.append(og_x)
		targets.append(og_y)
		#artificial time series signal
		if abs(addProb) > 1. or addProb < 0.:
			addProb = 1.
		if choice([True, False], p=[addProb,1.-addProb]):
			num_batches += 1
			inputs.append(normal(1,sigma)*og_x)
			if scale_y:
				inputs.append(normal(1,sigma)*og_y)
			else:
				targets.append(og_y)
	inputs = tf.data.Dataset.from_tensor_slices(inputs)
	targets = tf.data.Dataset.from_tensor_slices(targets)
	dts = tf.data.Dataset.zip((inputs,targets)).shuffle(buffer_size=num_batches)
	return dts, num_batches

def double(dataset, addProb=1.):
	inputs = []
	targets = []
	num_batches = 0
	for x,y in dataset:
		num_batches += 1
		og_x = x.numpy()
		og_y = y.numpy()
		inputs.append(og_x)
		targets.append(og_y)
		#artificial time series signal
		if abs(addProb) > 1. or addProb < 0.:
			addProb = 1.
		if choice([True, False], p=[addProb,1.-addProb]):
			num_batches += 1
			inputs.append(og_x)
			targets.append(og_y)
	inputs = tf.data.Dataset.from_tensor_slices(inputs)
	targets = tf.data.Dataset.from_tensor_slices(targets)
	dts = tf.data.Dataset.zip((inputs,targets)).shuffle(buffer_size=num_batches)
	return dts, num_batches
