#! /usr/bin/python3

from clr_callback import CyclicLR
from numba import njit
from neuralNets import buildNN, multInputsNN, multOutputsNN, praae_init, add_lregulirizer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, Nadam
from keras.losses import Huber, MeanSquaredError, SparseCategoricalCrossentropy
from numpy import newaxis as nax
from numpy import concatenate
from tensorflow import convert_to_tensor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as pl
from my_datahanddlers import pre_process, pos_process
from model_wrappers import my_wp
from tqdm import tqdm
#from numba import cuda

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
        wait=None,verbose=True,val_set=None):                    #ver se esse wait é nescessario para os tds
    nES = False if patience else True
    best_mse = None
    new_mse = None
    inputs = []
    targets = []
    turns_n_improv = -1
    num_batches = 0
    num_epochs = 0
    val_hist = list()
    if isinstance(train_dts, tuple):
        inputs,targets = train_dts
        num_batches = targets.shape[0]
    else:
        num_batches = len(train_dts)
        if verbose:
            print('Getting the number of batches')
        for x,y in tqdm(train_dts) if verbose else train_dts:
            num_batches += 1
            inputs.append(x.numpy()[nax,:])
            targets.append(y.numpy()[nax,:])
        inputs = concatenate(inputs,axis=0)
        targets = concatenate(targets,axis=0)
        inputs = convert_to_tensor(inputs)
        targets = convert_to_tensor(targets)
    base_lr,max_lr,mode = lr_vals
    steps_per_epoch = int(num_batches/batch_size*(1-val_size))#*g            #acho q vale definir isso
    clr_step_size = k*steps_per_epoch
    #tenho q dar uma lida nesse stepsize
    #é o step size, tenho q ajeitar isso
    #callbacks
    if mode == 'exp_range':
        clr = CyclicLR(base_lr=base_lr,max_lr=max_lr,mode=mode,step_size=clr_step_size,gamma=.99994)
    elif mode == 'on_plateau':#####################change
        clr = ReduceLROnPlateau(monitor='val_mse', factor=.5, patience=patience//3, cooldown=2, min_lr=base_lr)
        K.set_value(model.optimizer.learning_rate, max_lr)
    else:
        clr = CyclicLR(base_lr=base_lr,max_lr=max_lr,mode=mode,step_size=clr_step_size)
    checkpoint = ModelCheckpoint(saveAt, monitor='val_mse', verbose=verbose,
        save_best_only=True, mode='auto', save_freq='epoch')#period=1)
    cont_training = True
    if nES:
        while(cont_training):
            history = model.fit(inputs,targets,
                    batch_size=batch_size, callbacks=[clr,checkpoint],
                    epochs=no_epochs,validation_split=val_size,
                    verbose=verbose,shuffle=True, workers=10)
            val_hist += history.history['val_mse']
            if not best_mse:
                best_mse = min(val_hist)
                counted = len(val_hist)
            elif min(val_hist[:-no_epochs]) < best_mse:
                new_mse = min(val_hist)
                counted = sum(1 for i in val_hist[:-no_epochs] if i <= best_mse*(1. + 1e-2))#talvez mudar
            cont_training = True if counted >= int(no_epochs*.2) else False
            if new_mse:
                turns_n_improv = 0
                best_mse = new_mse
                new_mse = None
            else:
                turns_n_improv += 1
            cont_training = cont_training if turns_n_improv < 2 else False
        num_epochs = len(val_hist)
    else:
        if not wait and no_epochs < 150:
            no_epochs = 150
        earlystopping = EarlyStopping(monitor='val_mse',patience=patience)
        #earlystopping = EarlyStopping(monitor='val_mse', patience=patience)
        if wait:
            history = model.fit(inputs,targets,
                                batch_size=batch_size, callbacks=[clr,checkpoint],epochs=wait,
                                validation_split=val_size,
                                verbose=verbose,shuffle=True, workers=10)#,steps_per_epoch=steps_per_epoch)
            num_epochs = len(history.history['val_mse'])
            best_mse = min(history.history['val_mse'])
        while (cont_training):
            history = model.fit(inputs,targets,
                                batch_size=batch_size,
                                callbacks=[clr,earlystopping,checkpoint],epochs=no_epochs,
                                validation_split=val_size,
                                verbose=verbose,shuffle=True, workers=10)#,steps_per_epoch=steps_per_epoch)
            #check if relative change is greater then 10% of ref. value
            #|x - x_ref|/x_ref
            val_hist = history.history['val_mse']
            l_val = val_hist[-1]
            num_epochs += len(val_hist)
            if len(val_hist) < no_epochs:
                cont_training = False
            if cont_training and wait and min(val_hist) > best_mse:
                cont_training = False

#    del inputs
#    del targets
#    device = cuda.get_current_device()
#    device.reset()

    return num_epochs

def aeTrain(layer_config,batch_size,train_dts,lr_vals,optimizer=Adam,patience=3,k=4,no_epochs=25,
        saveAt='tmpAE.h5',checkLast=None,dif_tol=.10,val_size=.2,base_ae=None,verbose=True,wait=None,regs=None,
        val_set=None):
    #k is a constant that scale the step size
    #obs.: cycle = floor(1+iterations/(2*step_size))
    if isinstance(layer_config[0], list):
        if len(layer_config) == 2:
            ae = multInputsNN(layer_config[0], layer_config[1])
            #comment the line below if no wrapper will be used
            #ae = my_wp(layer_config[0][1], ae)#, pre_process, pos_process)#######modificacao
        else:
            inp, brns, out = layer_config
            ae, n_inp = multOutputsNN(inp, brns, return_layer_config=False)
            ae = buildNN(out, prev_layers=ae, return_layer_config=False)[0]
            ae = Model(inputs=n_inp, outputs=ae)
            #comment the line below if no wrapper will be used
            #ae = my_wp(layer_config[0][1], ae)#, pre_process, pos_process)#######modificacao
    else:
        ae = buildNN(layer_config)
        #comment the line below if no wrapper will be used
        #ae = my_wp(layer_config[0][1], ae)#, pre_process, pos_process)#######modificacao

    if regs and regs[0]:
        lays, reg = regs
        add_lregulirizer(ae, reg, lays)

    if base_ae:#################################################
        praae_init(ae,base_ae)

    if verbose:
        ae.summary()

    ae.compile(loss=MeanSquaredError(), optimizer=optimizer(), metrics=['mse','mae'])
    num_epochs = training(ae,batch_size,train_dts,lr_vals,patience=patience,k=k,no_epochs=no_epochs,
                saveAt=saveAt,checkLast=checkLast,dif_tol=dif_tol,val_size=val_size,verbose=verbose,wait=wait)
    del ae 
    return num_epochs
    #return training(ae,batch_size,train_dts,lr_vals,patience=patience,k=k,no_epochs=no_epochs,
    #            saveAt=saveAt,checkLast=checkLast,dif_tol=dif_tol,val_size=val_size,verbose=verbose,wait=wait)

def dnetTrain(layer_config,batch_size,train_dts,lr_vals,optimizer=Adam,patience=3,k=4,
    no_epochs=25,saveAt='tmpNeural.h5',checkLast=None,dif_tol=.10,val_size=.2,verbose=True):
    #k is a constant that scale the step size
    #obs.: cycle = floor(1+iterations/(2*step_size))
    d_network = buildNN(layer_config)##############################################################
    if verbose:
        d_network.summary()
    d_network.compile(loss=SparseCategoricalCrossentropy(), optimizer=optimizer(), metrics=['accuracy','mse'])
    num_epochs = training(d_network,batch_size,train_dts,lr_vals,patience=patience,k=k,no_epochs=no_epochs,
                saveAt=saveAt,checkLast=checkLast,dif_tol=dif_tol,val_size=val_size,verbose=verbose,wait=wait)
    del d_network
    return num_epochs
    #return training(d_network,batch_size,train_dts,lr_vals,patience=patience,k=k,no_epochs=no_epochs,
    #            saveAt=saveAt,checkLast=checkLast,dif_tol=dif_tol,val_size=val_size,verbose=verbose)
