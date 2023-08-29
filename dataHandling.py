#! /usr/bin/python3

import numpy as np
import tensorflow as tf
from numba import njit
import genDataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow import expand_dims, concat
import pandas as pd
import random
import concurrent.futures as concurrent

#getMin and getMax do the same as (min|max)(...,axis=0)
#the original are not supported by numba
@njit(nogil=True)
def getMin(matrix):
    num_cols = np.shape(matrix)[1]
    mins = np.zeros_like(matrix[0,:])
    for i in range(num_cols):
        mins[i] = np.min(matrix[:,i])
    return mins

@njit(nogil=True)
def getMax(matrix):
    num_cols = np.shape(matrix)[1]
    maxs = np.zeros_like(matrix[0,:])
    for i in range(num_cols):
        maxs[i] = np.max(matrix[:,i])
    return maxs

#@njit(nogil=True)
def stdDev(values,mean):
    denominator = float(len(values)-1)
    if denominator == 0.0:
        denominator = 1.0
    sumErr = 0.0
    for i in values:
        sumErr += (i-mean)**2
    return np.sqrt(sumErr/denominator)

@njit(nogil=True)
def scale(matrix,t_min=-1,t_max=1):        #t == target, as target range
    matrix_min = getMin(matrix)
    matrix_max = getMax(matrix)
    sMatrix = (matrix-matrix_min)/(matrix_max-matrix_min)
    sMatrix *= (t_max-t_min)
    return (sMatrix+t_min,matrix_min,matrix_max)
    #does not contemplate the case where the column is composed by
    #the same value. Return nan for all column

@njit(nogil=True)
def matrixSplit(matrix,where2start,where2end):
    stop = where2end-where2start
    maxLines = matrix.shape[0]
    removed = matrix[where2start:where2start+stop]
    if where2start != 0:
        res = matrix[0:where2start]
        res = np.concatenate((res,matrix[where2start+stop:maxLines]),axis=0)
    else:
        res = matrix[stop:maxLines]
    return (res,removed)

def simWindow(dataset, batch_size, window_shift):
    samples, targets = dataset
    if isinstance(batch_size,list):
        samp_batch = batch_size[0]
        tg_batch = batch_size[1]
    else:
        samp_batch = tg_batch = batch_size
    samples = tf.data.Dataset.from_tensor_slices(samples).window(size=samp_batch,
        shift=window_shift,drop_remainder=True)
    samples = samples.flat_map(lambda window: window.batch(samp_batch))
    targets = tf.data.Dataset.from_tensor_slices(targets).window(size=tg_batch,
        shift=window_shift,drop_remainder=True)
    targets = targets.flat_map(lambda window: window.batch(tg_batch))
    dts = tf.data.Dataset.zip((samples,targets))
    return dts

def setDataset(dataset, batch_size, window_shift=None):
    samples, targets = dataset
    if window_shift:
        return simWindow(dataset,batch_size,window_shift)
    if isinstance(batch_size,list):
        samp_batch = batch_size[0]
        tg_batch = batch_size[1]
    else:
        samp_batch = tg_batch = batch_size
    samples = tf.data.Dataset.from_tensor_slices(samples).batch(samp_batch)
    targets = tf.data.Dataset.from_tensor_slices(targets).batch(tg_batch)
    dts = tf.data.Dataset.zip((samples,targets))
    return dts

def scaleBatches(dataset,dRange=[0.,1.],s_ipts=True,s_tgs=False):
    t_min, t_max = dRange
    inputs = targets = None
    for x,y in dataset.take(1):
        ipts_size = tf.size(x).numpy()
        tgs_size = tf.size(y).numpy()
    for x,y in dataset:
        if s_ipts:
            x = scale(x.numpy()[:,np.newaxis],t_min,t_max)[0].T.squeeze()
        if s_tgs:
            y = scale(y.numpy()[:,np.newaxis],t_min,t_max)[0].T.squeeze()
        if isinstance(inputs,np.ndarray):
            inputs = np.concatenate((inputs, x))
        else:
            if not isinstance(x,np.ndarray):
                inputs = x.numpy()
            else:
                inputs = x
        if isinstance(targets,np.ndarray):
            targets = np.concatenate((targets, y))
        else:
            if not isinstance(y,np.ndarray):
                targets = y.numpy()
            else:
                targets = y
    return setDataset((inputs,targets), [ipts_size,tgs_size])

#def initDataset(dts_loc,cols,smp_size,test_size,wnd_stride,test_stride,wnd_str_comp=None,combine=None):
def initDataset(dts_loc,cols,smp_size,test_size,wnd_stride,test_stride,wnd_str_comp=None):
    #generate datasets -- specific to my datasets
    if isinstance(dts_loc,list) and len(dts_loc) == 2:
        train_loc, test_loc = dts_loc
        labels,raw,trainSet = genDataset.csvFl(train_loc,cols=cols)
        labels,raw,testSet = genDataset.csvFl(test_loc,cols=cols)
    else:
        labels,dts_raw,dataset = genDataset.csvFl(dts_loc,cols=cols)
        X_train,X_test,Y_train,Y_test = train_test_split(dataset[:,0],dataset[:,1],test_size=test_size,shuffle=False)
        trainSet = np.concatenate((X_train[:,np.newaxis],Y_train[:,np.newaxis]),axis=1)
        testSet = np.concatenate((X_test[:,np.newaxis],Y_test[:,np.newaxis]),axis=1)
        for aux in [X_train,X_test,Y_train,Y_test]:
            del aux
    #generate shift in training dataset and test dataset
    train_dts = setDataset((trainSet[:,1],trainSet[:,1]), smp_size,
        window_shift=wnd_stride)
    if wnd_str_comp:
        trn_dts = [train_dts]
        for wnd_str in wnd_str_comp:
            trn_dts.append(setDataset((trainSet[:,1],trainSet[:,1]), smp_size,
                window_shift=wnd_str))
        x_vals = []
        y_vals = []
        for dts in trn_dts:
            for x,y in dts:
                x_vals += x.numpy().tolist()
                y_vals += y.numpy().tolist()
        train_dts = setDataset((x_vals,y_vals), smp_size)
    test_dts = setDataset((testSet[:,0],testSet[:,1]), smp_size,
        window_shift=test_stride)
    num_tr_bat = 0
    for x in train_dts: num_tr_bat += 1
    num_ts_bat = 0
    for x in test_dts: num_ts_bat += 1
    return (train_dts,num_tr_bat), (test_dts,num_ts_bat)

class get_samples(Sequence):
    def __init__(self, dataset,
                       batch_size,
                       strides=0,
                       cont_diff=False,
                       #return_time=False,
                       #act_range=(0.,1.),
                       snrs=None,
                       p_apply = 0.,
                       encode=None,
                       samples=None,
         #              B_size=1,
                       ):#, device=None):
        assert batch_size > 0
        self.dataset = dataset
        length = -1 #ignore header   -- when reading dataset, add this
        with open(dataset, 'r') as dt_file:
            for _ in dt_file:
                length += 1
        self.batch_size = batch_size
 
        if isinstance(strides, int):
            strides = [strides]
        self.strides = strides

        if isinstance(samples, list):
            self.samples = samples
        else:
            samples = list()
            for stride in strides:
                if stride <= 0:
                    stride = batch_size
                aux = list(range(1,length,stride))
                while aux[-1] + self.batch_size > length:
                    aux.pop()
                samples += aux
            self.samples = list(dict.fromkeys(samples))     #remove duplicates

        #self.B_size = B_size
        np.random.shuffle(self.samples)
        #self.pivots = list(range(0,len(self.samples),B_size))
        #transforms
        #self.normalizer = map_to(*act_range)

        #self.labels = labels
        self.cont_diff = cont_diff
        #self.return_time = return_time
        #self.device = device
        self.p_apply = p_apply if 0. < p_apply <= 1. else 0.
        self.snrs=snrs if isinstance(snrs,list) or isinstance(snrs,tuple) else [snrs]
        if len(self.snrs) == 1 and self.snrs[0] == 0:
            self.p_apply == 0.
        self.encode = encode

    def subset(self, indices):
        return get_samples(self.dataset, self.batch_size, self.strides,
                            self.cont_diff, self.snrs, self.p_apply, self.encode, indices)

    #define indices
    def get_indices(self,split_ratio=0.):
        indices = list(range(self.__len__()))
        if 0. < split_ratio < 1.:
            from random import shuffle
            shuffle(indices)
            n = int((1. - split_ratio)*len(indices))
            return indices[:n], indices[n:]
        return indices,[]  

    # adapted from Mathuranathan Viswanathan (gaussianwaves.com)\
    # https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function
    #input must be list
    def __awgn(self,s,SNRdB): #for 1-d signal
        """
        AWGN channel
        Add AWGN noise to input signal. The function adds AWGN noise vector to signal 's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
        returns the noise vector 'n' to be added to the signal 's' and the power spectral density N0 of noise added
        Parameters:
            s : input/transmitted signal vector
            SNRdB : desired signal to noise ratio (expressed in dB) for the received signal
        Returns:
            n : noise vector
        """
        gamma = 10**(SNRdB/10) #SNR to linear scale
        P=sum(abs(s)**2)/s.size

        N0=P/gamma # Find the noise spectral density

        return np.float32(s + np.sqrt(N0/2)*np.random.standard_normal(s.shape))

    def __len__(self):
        #return len(self.pivots)
        return len(self.samples)# // self.B_size

    def on_epoch_end(self):
        np.random.shuffle(self.samples)

#   def __getitem__(self, index):
#       #pivot = self.pivots[index]
#       X,Y = list(),list()
#       with concurrent.ThreadPoolExecutor() as executor:
#           samples = executor.map(self._get_sample, self.samples[index:index+self.B_size])
#           for x,y in samples:
#               X.append(x)
#               Y.append(y)
#       X = concat(X, 0)
#       Y = concat(Y, 0)
#       return X,Y

    #def _get_sample(self, index):
    def __getitem__(self, index):
        i = self.samples[index]
        if self.cont_diff and index != 1:
            sample = pd.read_csv(self.dataset, header=None, skiprows=i-1,
                                nrows=self.batch_size+1, dtype={1: np.float32})
        else:
            sample = pd.read_csv(self.dataset, header=None, skiprows=i,
                      nrows=self.batch_size, dtype={1: np.float32})
        #t_stamps = sample.iloc[:,0]
        raw_tmps = sample.iloc[:,1]

        #prepare sample with dpcm like transform
        #first_tmp = raw_tmps[0]                                                 #get first sample
        raw_tmps = raw_tmps.diff().fillna(0)[-self.batch_size:].values
        if 0. < self.p_apply <= 1.:
            if self.p_apply == 1. or random.random() > 1 - self.p_apply:
                tmps = self.__awgn(raw_tmps, np.random.choice(self.snrs) if len(self.snrs) > 1 else self.snrs[0])
            else:
                tmps = raw_tmps
        else:
            tmps = raw_tmps#.diff().fillna(0)[-self.batch_size:]                 #compute differences

        if self.encode:
            tmps = self.encode(tmps)[0]
            raw_tmps = self.encode(raw_tmps)[0]

        #prepare tensors
        raw_tmps = tf.convert_to_tensor(raw_tmps)
        tmps = tf.convert_to_tensor(tmps)

        return tmps, raw_tmps

if __name__ == '__main__':
    matrix = []; targ = []
    for i in range(100):
        line = []
        for j in range(5):
            line.append(i)
        targ.append(i)
        matrix.append(line)
    matrix = np.array(matrix)
    targ = np.array(targ)[:,np.newaxis]

    BATCH_SIZE = 20
    
    (train_smp,train_tg), num_samples = duplicator(matrix,y=targ,at_each=BATCH_SIZE, num_lines=4)
    (train_dts,valid_dts) = matrixSplit(train_smp, num_samples-(2*BATCH_SIZE), num_samples)
    (train_tg,val_tg) = matrixSplit(train_tg, num_samples-(2*BATCH_SIZE), num_samples)

    train_dts = tf.data.Dataset.from_tensor_slices((train_dts,train_tg)).batch(20)
    valid_dts = tf.data.Dataset.from_tensor_slices((valid_dts,val_tg)).batch(20)

    num_batches = 0
    print('Training Dataset')
    for features, labels in train_dts:
        print(features.numpy()); print(labels.numpy())
        num_batches += 1
    print('Number of Batches: {}'.format(num_batches))

    num_batches = 0
    print('Validation Dataset')
    for features, labels in valid_dts:
        print(features.numpy()); print(labels.numpy())
        num_batches += 1
    print('Number of Batches: {}'.format(num_batches))

