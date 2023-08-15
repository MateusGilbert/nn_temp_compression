#! /usr/bin/python3

from numpy import newaxis as nax, ndarray, sqrt
from numpy.random import normal, choice
from tensorflow import concat
from numpy import flip
import tensorflow as tf
from numpy.random import standard_normal
from tqdm import tqdm

#sigma is a hyperparameter
#tem que conferir se não está fazendo merda no shuffle

def jittering(dataset, sigma=.5, prob=None, addProb=1.):
    inputs = []
    targets = []
    num_batches = 0
    print('Data Augmentation: Jittering')
    for x,y in tqdm(dataset):
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

# adapted from Mathuranathan Viswanathan (gaussianwaves.com)\
# https://www.gaussianwaves.com/2015/06/how-to-generate-awgn-noise-in-matlaboctave-without-using-in-built-awgn-function
    
def awgn(s,SNRdB): #for 1-d signal
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
    is_tensor = False
    if isinstance(s,list): #Actual power in the vector
        P=sum(list(map(lambda x: abs(x)**2, s)))/len(s)
    elif isinstance(s, ndarray):
        P=sum(abs(s)**2)/s.size 
    else:
        aux = s.numpy()
        P=sum(abs(aux)**2)/aux.size 
        is_tensor = True
    
    N0=P/gamma # Find the noise spectral density
    if is_tensor:
        return tf.convert_to_tensor(sqrt(N0/2)*standard_normal(aux.shape))
    return sqrt(N0/2)*standard_normal(s.shape)

def jittering_awgn(dataset, sigma=.5, prob=None, addProb=1.):
    inputs = []
    targets = []
    num_batches = 0
    print('Data Augmentation: Jittering (dB)')
    for x,y in tqdm(dataset):
        num_batches += 1
        og_x = x.numpy()
        og_y = y.numpy()
        inputs.append(og_x)
        targets.append(og_y)
        #artificial time series signal
        if abs(addProb) > 1. or addProb < 0.:
            addProb = 1.
        if addProb == 1. or choice([True, False], p=[addProb,1.-addProb]):
            if isinstance(sigma,list):# and not isinstance(addProb,int):
                if prob == None:
                    prob = [1/len(sigma) for i in range(len(sigma))]

                if not isinstance(prob,list):########gambiarra
                    for s in sigma:
                        num_batches += 1
                        inputs.append(og_x + awgn(og_x, s))
                        targets.append(og_y)
                else:
                    s_aux = choice(sigma,1,p=prob).squeeze()
                    num_batches += 1
                    inputs.append(og_x + awgn(og_x,s_aux))
                    targets.append(og_y)
            else:
                num_batches += 1
                inputs.append(og_x + awgn(og_x,sigma))
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
    print('Data Augmentation: Flipping')
    for x,y in tqdm(dataset):
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
    print('Data Augmentation: Scaling')
    for x,y in tqdm(dataset):
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
            inputs.append(normal(1.,sigma)*og_x)
            if scale_y:
                inputs.append(normal(1.,sigma)*og_y)
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
    print('Data Augmentation: Double')
    for x,y in tqdm(dataset):
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
