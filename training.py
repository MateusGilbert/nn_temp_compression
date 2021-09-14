#! /usr/bin/python3

from clr_callback import CyclicLR
from numba import njit
from neuralNets import buildNN, buildNN2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, Nadam
from numpy import newaxis as nax
#from tensorflow import concat
from numpy import concatenate
from tensorflow import convert_to_tensor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as pl

from keras import backend as K
from keras.callbacks import TensorBoard

class LRTensorBoard(TensorBoard):
	def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
		super().__init__(log_dir=log_dir, **kwargs)

	def on_epoch_end(self, epoch, logs=None):
		logs = logs or {}
		logs.update({'lr': K.eval(self.model.optimizer.lr)})
		super().on_epoch_end(epoch, logs)

def training(model,batch_size,train_dts,lr_vals,optimizer=Adam,patience=3,k=4,no_epochs=25,
		saveAt='tmpAE.h5',checkLast=None,dif_tol=.10,val_size=.2,shuffle=True,
		wait=None,verbose=True):					#ver se esse wait é nescessario para os tds
	nES = False if patience else True
	best_mse = None
	new_mse = None
	inputs = []
	targets = []
	turns_n_improv = -1
	num_batches = 0
	num_epochs = 0
	val_hist = list()
	for x,y in train_dts:
		num_batches += 1
		inputs.append(x.numpy()[nax,:])
		targets.append(y.numpy()[nax,:])
	inputs = concatenate(inputs,axis=0)
	targets = concatenate(targets,axis=0)
	#inputs,val_inp,targets,val_tar = train_test_split(inputs,targets,test_size=val_size,shuffle=False)
	inputs = convert_to_tensor(inputs)
	#val_inp = convert_to_tensor(val_inp)
	targets = convert_to_tensor(targets)
	#val_tar = convert_to_tensor(val_tar)
	base_lr,max_lr,mode = lr_vals
	steps_per_epoch = int(num_batches/batch_size*(1-val_size))#*g			#acho q vale definir isso
	clr_step_size = k*steps_per_epoch
	#tenho q dar uma lida nesse stepsize
	#é o step size, tenho q ajeitar isso
	#callbacks
	if mode == 'exp_range':
		clr = CyclicLR(base_lr=base_lr,max_lr=max_lr,mode=mode,step_size=clr_step_size,gamma=.99994)
	else:
		clr = CyclicLR(base_lr=base_lr,max_lr=max_lr,mode=mode,step_size=clr_step_size)
	checkpoint = ModelCheckpoint(saveAt, monitor='val_mse', verbose=verbose,
		save_best_only=True, mode='auto', period=1)
	cont_training = True
	if nES:
		while(cont_training):
			history = model.fit(inputs,targets,batch_size=batch_size,
				callbacks=[clr,checkpoint],epochs=no_epochs,validation_split=val_size,verbose=verbose,shuffle=True)
				#callbacks=[clr,checkpoint,LRTensorBoard(log_dir='/tmp/tb_log')],epochs=no_epochs,validation_split=val_size,verbose=verbose,shuffle=True)
			val_hist += history.history['val_mse']
			if not best_mse:
				best_mse = min(val_hist)
				counted = len(val_hist)
			elif min(val_hist[:-no_epochs]) < best_mse:
				new_mse = min(val_hist)
				counted = sum(1 for i in val_hist[:-no_epochs] if i <= best_mse*(1. + 1e-2))#talvez mudar
			#op = input('Continue training? [Y]es/[N]o: ')
			cont_training = True if counted >= int(no_epochs*.2) else False
			if new_mse:
				turns_n_improv = 0
				best_mse = new_mse
				new_mse = None
			else:
				turns_n_improv += 1
			cont_training = cont_training if turns_n_improv < 2 else False
			#model = load_model(saveAt)
			#model.compile(loss='mse', optimizer=optimizer(), metrics=['mse','mae'])		#trocar tds Adam por optimizer 
		num_epochs = len(val_hist)
	else:
		if not wait and no_epochs < 150:
			no_epochs = 150
		earlystopping = EarlyStopping(monitor='val_mse', patience=patience)
#		if not checkLast:
#			checkLast = int(no_epochs/2)
		if wait:
			history = model.fit(inputs,targets,batch_size=batch_size,
				callbacks=[clr,checkpoint],epochs=wait,#validation_data=(val_inp,val_tar),
				#callbacks=[clr,checkpoint,LRTensorBoard(log_dir='/tmp/tb_log')],epochs=int(3/4*no_epochs),#validation_data=(val_inp,val_tar),
				validation_split=val_size,verbose=verbose,shuffle=True)#,steps_per_epoch=steps_per_epoch)
			num_epochs = len(history.history['val_mse'])
			best_mse = min(history.history['val_mse'])
		while (cont_training):
			history = model.fit(inputs,targets,batch_size=batch_size,
				callbacks=[clr,earlystopping,checkpoint],epochs=no_epochs,#validation_data=(val_inp,val_tar),
				#callbacks=[clr,earlystopping,checkpoint,LRTensorBoard(log_dir='/tmp/tb_log')],epochs=no_epochs,#validation_data=(val_inp,val_tar),
				validation_split=val_size,verbose=verbose,shuffle=True)#,steps_per_epoch=steps_per_epoch)
			#check if relative change is greater then 10% of ref. value
			#|x - x_ref|/x_ref
			val_hist = history.history['val_mse']
			l_val = val_hist[-1]
			#ref_val = val_hist[-checkLast if len(val_hist) > checkLast else 0]
			num_epochs += len(val_hist)
			if len(val_hist) < no_epochs:
				cont_training = False
#			if cont_training and abs(l_val-ref_val)/ref_val < dif_tol:
#				cont_training = False
			if cont_training and wait and min(val_hist) > best_mse:
				cont_training = False
	return num_epochs

#def aeTrain(layers,batch_size,train_dts,spec,lr_vals,out_act='sigmoid',optimizer=Adam,dropout=None,patience=3,k=4,no_epochs=25,
#		saveAt='tmpAE.h5',checkLast=None,dif_tol=.10,val_size=.2,verbose=True,addBatchNorm=(False,False)):
def aeTrain(layer_config,batch_size,train_dts,lr_vals,optimizer=Adam,patience=3,k=4,no_epochs=25,
		saveAt='tmpAE.h5',checkLast=None,dif_tol=.10,val_size=.2,verbose=True,wait=None):
	#k is a constant that scale the step size
	#obs.: cycle = floor(1+iterations/(2*step_size))
#	spec = {'activation': act,
#				'initializer': 'glorot_uniform'}
	#smp_size,cmp_size = sizes
	#ae = buildNN(layers,spec,output=act,dropout=dropout,ae=(True,True))
#	if dropout:
#		ae = buildNN(layers,spec,dropout=dropout.copy(),output=out_act,addBatchNorm=addBatchNorm,ae=(True,True))
#	else:
#		ae = buildNN(layers,spec,output=out_act,addBatchNorm=addBatchNorm,ae=(True,True))
	ae = buildNN2(layer_config)
	if verbose:
		ae.summary()
	ae.compile(loss='mse', optimizer=optimizer(), metrics=['mse','mae'])
	return training(ae,batch_size,train_dts,lr_vals,patience=patience,k=k,no_epochs=no_epochs,
				saveAt=saveAt,checkLast=checkLast,dif_tol=dif_tol,val_size=val_size,verbose=verbose,wait=wait)

#def dnetTrain(layers,out_act,batch_size,train_dts,spec,lr_vals,optimizer=Adam,dropout=None,patience=3,k=4,
#	no_epochs=25,saveAt='tmpNeural.h5',checkLast=None,dif_tol=.10,val_size=.2,wait=False,verbose=True,addBatchNorm=False):
def dnetTrain(layer_config,batch_size,train_dts,lr_vals,optimizer=Adam,patience=3,k=4,
	no_epochs=25,saveAt='tmpNeural.h5',checkLast=None,dif_tol=.10,val_size=.2,verbose=True):
	#k is a constant that scale the step size
	#obs.: cycle = floor(1+iterations/(2*step_size))
#	if dropout:
#		d_network = buildNN(layers,spec,dropout=dropout.copy(),output=out_act,addBatchNorm=addBatchNorm)
#	else:
#		d_network = buildNN(layers,spec,output=out_act,addBatchNorm=addBatchNorm)
	d_network = buildNN2(layer_config)
	if verbose:
		d_network.summary()
	d_network.compile(loss='mse', optimizer=optimizer(), metrics=['mse','mae'])
	return training(d_network,batch_size,train_dts,lr_vals,patience=patience,k=k,no_epochs=no_epochs,
				saveAt=saveAt,checkLast=checkLast,dif_tol=dif_tol,val_size=val_size,verbose=verbose)
