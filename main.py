#! /usr/bin/python3

from t2_taxonomy import t2_taxonomy as t2
from neuralNets import custom_sigmoid
from reader import get_lr
from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from numpy.random import seed

#####
seed(1671)
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#tf.keras.backend.tensorflow_backend.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)
get_custom_objects().update({'custom_sigmoid': custom_sigmoid})
#####
func = t2
func_str = 'Asymmetric Autoencoders'#'AE Taxonomy'
print('Testing: {}.'.format(func_str))
dirname = func()
print('Testing finished successfully! Check results at {}.'.format(dirname))
