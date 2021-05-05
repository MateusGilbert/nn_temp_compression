#! /usr/bin/python3

from numpy import newaxis as nax
#from numpy.random import normal, choice
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import Mean
from dataHandling import scale
from tensorflow import transpose, convert_to_tensor
from sklearn.metrics import mean_squared_error as MSE

#dps acrescentar random noise
def ae_test_dts(ae_model, dataset, func_range, test_stride, verbose=True):
	outputs = []
	test_loss_avg = Mean()
	for x,y in dataset:
		(y_scaled,mins,maxs) = scale(y.numpy()[:,nax],func_range[0],func_range[1])
		#conversion to np.array of shape (1,n) because of njit functions (see dataHandling.py)
		#output = ae_model(y_scaled[nax,:],training=False)
		output = ae_model(y_scaled.T,training=False)
		output = (scale(output.numpy().T,mins,maxs)[0])[:test_stride]
		outputs += (output.T.squeeze()).tolist()
		output = convert_to_tensor(output)
		y_ref = y.numpy().tolist()[:test_stride]
		y_out = output.numpy().tolist()[:test_stride]
		loss = MSE(y_true=y_ref,y_pred=y_out)
		if loss < 50:#para tirar o outlier do NaN
			test_loss_avg.update_state(loss)
		if verbose:
			print('Train Loss: {}'.format(loss))
	return outputs, test_loss_avg.result().numpy()

def dec_test_dts(encoder, dec_model, dataset, func_range, test_stride, verbose=True, scale_tg=True):
	outputs = []
	test_loss_avg = Mean()
	for x,y in dataset:
		(y_scaled,mins,maxs) = scale(y.numpy()[:,nax],func_range[0],func_range[1])
		#conversion to np.array of shape (1,n) because of njit 
		#functions (see dataHandling.py)
		y_scaled = y_scaled.T
		y_enc = encoder.encode(y_scaled)
		if isinstance(dec_model,type(Sequential())):
			output = dec_model(y_enc,training=False)
		else:
			output = dec_model.decode(y_enc)#.numpy().squeeze()
		if scale_tg:
			output = scale(output.numpy().T,mins,maxs)[0][:test_stride]
		outputs += (output.T.squeeze()).tolist()
		output = convert_to_tensor(output)
		y_ref = y.numpy().tolist()[:test_stride]
		y_out = output.numpy().squeeze().tolist()[:test_stride]
		loss = MSE(y_true=y_ref,y_pred=y_out)
		if loss < 50:#para tirar o outlier do NaN
			test_loss_avg.update_state(loss)
		if verbose:
			print('Train loss: {}'.format(loss))
	return outputs, test_loss_avg.result().numpy()
