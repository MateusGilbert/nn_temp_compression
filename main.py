#! /usr/bin/python3

from new_taxonomy import t2_taxonomy as t2
#from t2_taxonomy import t2_taxonomy as t2
#from t3_taxonomy import t3_taxonomy as t3
#from t4_taxonomy import t4_taxonomy as t4
#from t_dimred import t_dimred as t5
from neuralNets import custom_sigmoid
from reader import get_lr_2
#from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf
from numpy.random import seed
from datetime import datetime
import os

#define log file
now = datetime.now()
log_file = f'log_{now.strftime("%m_%d_%H_%M")}.txt'
#####
seed(1671)
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
tf.config.run_functions_eagerly(True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#tf.keras.backend.tensorflow_backend.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)
#get_custom_objects().update({'custom_sigmoid': custom_sigmoid})
#####
func = t2
#func_str = 'AAEs for Classification' #'Asymmetric Autoencoders'# w/ Pseudo-Recursions'#'AE Taxonomy'
mstr = 'Testing: DPCM + AAEs\n'
if log_file:
    with open(log_file, 'w') as fd:
        fd.write(mstr)
else:
    print(mstr)

lr_vals = get_lr_2('lr_0205_modw.csv')#(125,25),'sigmoid','selu', other_sp=None)
cols = [0,1] #[f'T_{i}acc' for i in ['x']]#, 'y', 'z']] 
dirname = func(dir_id=f'aae_results_', cols=cols, verbose=True, lr_list=lr_vals, log_file=os.path.join(os.getcwd(),log_file))

mstr = f'Testing finished successfully! Check results at {dirname}.\n'
if log_file:
    with open(log_file,'a') as fd:
        fd.write(mstr)
else:
    print(mstr)
