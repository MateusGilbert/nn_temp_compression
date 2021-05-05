#! /usr/bin/python3

#OBS.: Training and testing loops are done in such a manner as to
# satisfy project requirements

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import MeanSquaredError as MSE
#from tensorflow import GradientTape
from dataHandling import setDataset, scale
from numpy import newaxis as nax
from numpy import concatenate, array, shape
from numba import njit
from time import time

@tf.function(experimental_compile=True)
def loss(model,inputs,targets,loss_obj=MSE(),training=False):
	#training = training is needed only if there are layers with different
	#behavior during training versus inference (e.g. Dropout)
	outputs = model(inputs, training=training)
	return loss_obj(y_true=targets,y_pred=outputs)

@tf.function(experimental_compile=True)
def grad(model,inputs,targets):
	with tf.GradientTape() as tape:
		loss_value = loss(model, inputs, targets, training=True)
	return loss_value, tape.gradient(loss_value, model.trainable_variables)

#even though it is a list, conversion is needed
#because njit does not work well with numba
@njit(nogil=True)
def stopChecker(entries, epsilon=.1):
	l_range = shape(entries)[0]
	for i in range(1,l_range):
		if entries[i-1] < entries[i]*(1.+epsilon):
			return True
	return False

#same requirement (np.array instead of list) as the function above
#@njit(nogil=True)
#def ZigZagChk(entries, tolerance=10):
#	l_range = shape(entries)[0]
#	count = 0
#	for i in range(1,l_range):
#		if count % 2:
#			if entries[i-1] > entries[i]:
#				count += 1
#		else:
#			if entries[i-1] < entries[i]:
#				count += 1
#		if count == tolerance:
#			return False
#	return True

@njit(nogil=True)
def CheckProgress(entries, epsilon=.25):
	l_range = shape(entries)[0]
	if abs(entries[0] - entries[int(l_range/4)]) > epsilon*entries[int(l_range/4)]:
		return True
	if abs(entries[0] - entries[int(l_range/2)]) > epsilon*entries[int(l_range/2)]:
		return True
	if abs(entries[0] - entries[l_range]) > epsilon*entries[int(l_range)]:
		return True
	return False

#Training earlystoping
#@tf.function(experimental_compile=True)
def training(model, dataset, optimizer=Adam(learning_rate=.001), train_style='fixed',
	verbose=False, batch_size=10, batches_val=.2, patience=2, ch_range=100,
	startChecking=100, saveAt='my_model.h5', scaleTo=None,shuffle=False):
	#if datarounding, scaleTo = [min,max] (see scale function @ dataHandling.py)
	samp_num = 0
	for x in dataset:
		samp_num += 1
	train_samp = int((1.-batches_val)*samp_num)
	if shuffle:
		aux_dts = dataset.shuffle(buffer_size=samp_num)
		train_dts = aux_dts.take(train_samp)
		valid_dts = aux_dts.skip(train_samp)
	else:
		train_dts = dataset.take(train_samp)
		valid_dts = dataset.skip(train_samp)
	if batch_size > 1:
		train_dts = train_dts.batch(batch_size)

	#initialize measurement variables
	epoch = 0
	min_val_loss = 100.
	train_loss_results = []
	val_loss_results = []
	contTraining = True
	#initialize zig-zag par.
	checkRange = ch_range
	checkAtEach = int(ch_range/4)
	while contTraining:
		if verbose:
			start = time()
		epoch += 1
		#for i in range(iterations):
		train_loss_avg = Mean()
		val_loss_avg = Mean()
#			if train_style == 'cross-validation':
#				train_dts, valid_dts = setSets(samples, labels, i, i+num_bVal, batch_size, difSize=difSize,
#					it_aux=[num_batches-num_bVal,num_bVal-1])

		#training
		for inputs, targets in train_dts:
			if scaleTo:
				if inputs.numpy().shape == targets.numpy().shape:	#we are dealing with ae
					inputs = tf.transpose(
						tf.convert_to_tensor(scale(inputs.numpy().T,scaleTo[0],scaleTo[1])[0]))
				targets = tf.transpose(
						tf.convert_to_tensor(scale(targets.numpy().T,scaleTo[0],scaleTo[1])[0]))
			#optimize the model
			loss_value, grads = grad(model, inputs, targets)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			#track progress
			train_loss_avg.update_state(loss_value)
		#validation
		for inputs, targets in valid_dts:
			if scaleTo:
				if inputs.numpy().shape == targets.numpy().shape:	#we are dealing with ae
					inputs = tf.transpose(tf.convert_to_tensor(
								scale(inputs.numpy()[:,nax],scaleTo[0],scaleTo[1])[0]))
				else:
					inputs = tf.convert_to_tensor([inputs.numpy()])
				targets = tf.transpose(tf.convert_to_tensor(
								scale(targets.numpy()[:,nax],scaleTo[0],scaleTo[1])[0]))
			val_loss_avg.update_state(loss(model,inputs,targets))

		#update results
		train_loss_results.append(train_loss_avg.result())
		val_loss_results.append(val_loss_avg.result().numpy())
		if verbose:
			print('Epoch {:03d}: Loss: {:g}; Val. Loss: {:g}'.format(epoch,
																			train_loss_avg.result(),
																			val_loss_avg.result()))
		#verify stoping criteria
		if len(val_loss_results) > patience:
			val_loss_results.reverse()
			contTraining = stopChecker(array(val_loss_results[:patience+1]))
			if contTraining and (epoch > startChecking) and (not epoch % checkAtEach):
				contTraining = CheckProgress(array(val_loss_results[:checkRange]))
			val_loss_results.reverse()
		#save if achieved the best validation loss so far
		if val_loss_results[epoch-1] < min_val_loss:
			min_val_loss = val_loss_results[epoch-1]
			model.save(saveAt)
		if verbose:
			stop = time()
			print('It took: {:.3f}s'.format(stop-start))
	return epoch
