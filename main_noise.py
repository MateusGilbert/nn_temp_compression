#! /usr/bin/python3

from new_taxonomy import t2_taxonomy as func
from reader import get_lr_2
import tensorflow as tf
from numpy.random import seed
from datetime import datetime
import os

#Sample/Compression##########################
smp_size = 100
cmp_size = 25
#AE specifications###########################
out_func = 'sigmoid'#'tanh'
hid_act = 'selu'#'relu'

models = [
    #basic AEs
#    ('AE-0', [('in', (smp_size,), None), ('dl', cmp_size, hid_act), ('dl', smp_size, out_func)]),
#    ('AE-1', [('in', (smp_size,), None), ('dl', 50, hid_act), ('dl', cmp_size, hid_act), ('dl', 50, hid_act), ('dl', smp_size, out_func)]),
#    ('AE-2', [('in', (smp_size,), None), ('dl', 75, hid_act), ('dl', 50, hid_act), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act), ('dl', 75, hid_act), ('dl', smp_size, out_func)]),
#    ('AE-3', [('in', (smp_size,), None), ('dl', 85, hid_act), ('dl', 65, hid_act), ('dl', 45, hid_act), ('dl', cmp_size, hid_act),
#                ('dl', 45, hid_act), ('dl', 65, hid_act), ('dl', 85, hid_act), ('dl', smp_size, out_func)]),
#    ('AE-4', [('in', (smp_size,), None), ('dl', 85, hid_act), ('dl', 70, hid_act), ('dl', 55, hid_act), ('dl', 40, hid_act), ('dl', cmp_size, hid_act),
#                ('dl', 40, hid_act), ('dl', 55, hid_act), ('dl', 70, hid_act), ('dl', 85, hid_act), ('dl', smp_size, out_func)]),

    #AAEs
#   ('AAE-1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act), ('dl', 50, hid_act), ('dl', smp_size, out_func)]),
#   ('AAE-2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#               ('dl', 50, hid_act), ('dl', 75, hid_act), ('dl', smp_size, out_func)]),
#   ('AAE-3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#               ('dl', 45, hid_act), ('dl', 65, hid_act), ('dl', 85, hid_act), ('dl', smp_size, out_func)]),
#   ('AAE-4', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#               ('dl', 40, hid_act), ('dl', 55, hid_act), ('dl', 70, hid_act), ('dl', 85, hid_act), ('dl', smp_size, out_func)]),

   #CAAEs
#  ('NCAAE-1.1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                 ('rs', (cmp_size, 1), None), ('up_conv', (4, 3, 2, 4), hid_act), ('up_conv', (4, 3, 2, 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-1.1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size,1), 4, 3, 2, 4), hid_act), ('up_conv', (4, 3, 2, 4), hid_act),
#                ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-2.1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 40, hid_act), ('rs', (40, 1), None), ('d_conv', ((40,1), 4, 3, 2, 4, 50), hid_act), ('up_conv', (4, 3, 2, 4), hid_act),
#                ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-1.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                 ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 4, 3, 2, 4), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [4, 4, 4]), hid_act),
#                 ('up_conv', (8, 3, 2, 4), hid_act),
#                 ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-2.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                 ('dl', 40, hid_act), ('rs', (40, 1), None), ('d_conv', ((40,1), 4, 3, 2, 4, 50), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [4, 4, 4]), hid_act),
#                 ('up_conv', (8, 3, 2, 4), hid_act),
#                 ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-1.3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size,1), 8, 3, 2, 4), hid_act), ('up_conv', (8, 3, 2, 8), hid_act),
#                ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-2.3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 40, hid_act), ('rs', (40, 1), None), ('d_conv', ((40,1), 8, 3, 2, 4, 40), hid_act), ('up_conv', (8, 3, 2, 8), hid_act),
#                ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-1.4', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                 ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 8, 3, 2, 4), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [8, 8, 8]), hid_act),
#                 ('up_conv', (8, 3, 2, 8), hid_act),
#                 ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
   ('CAAE-3.1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
                 ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size,1), 4, 3, 2, 4, [1.6]), hid_act), ('up_conv', (4, 3, 2, 4), hid_act),
                 ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
   ('CAAE-3.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 4, 3, 2, 4, [1.6]), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [4, 4, 4]), hid_act),
                  ('up_conv', (8, 3, 2, 4), hid_act),
                  ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-3.3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size,1), 8, 3, 2, 4, [1.6]), hid_act), ('up_conv', (8, 3, 2, 8), hid_act),
#                ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#  ('CAAE-3.4', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                 ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 8, 3, 2, 4, [1.6]), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [8, 8, 8]), hid_act),
#                 ('up_conv', (8, 3, 2, 8), hid_act),
#                 ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
]

#define log file
train_label = 'new_caaes_3_1'
now = datetime.now()
log_file = f'log_noise_{train_label}.txt'#{now.strftime("%m_%d_%H_%M")}.txt'
#####
seed(1671)
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
tf.config.run_functions_eagerly(True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#tf.keras.backend.tensorflow_backend.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)
#get_custom_objects().update({'custom_sigmoid': custom_sigmoid})
#####
mstr = 'Testing: new CAAEs + noise\n'
if log_file:
    with open(log_file, 'w') as fd:
        fd.write(mstr)
else:
    print(mstr)

lr_vals = None# get_lr_2('new_caaes.csv')#(125,25),'sigmoid','selu', other_sp=None)
cols = [0,1] #[f'T_{i}acc' for i in ['x']]#, 'y', 'z']] 
og_dir = os.getcwd()
dirname = func(models=models, snrs=[80, 60, 40, 30, 20, 15, 10, 5, 2.5], p_apply=1., dir_id=f'{train_label}_results_', cols=cols, verbose=False, lr_list=lr_vals, log_file=os.path.join(og_dir,log_file), gpu='/GPU:1')

os.chdir(og_dir)
mstr = f'Testing finished successfully! Check results at {dirname}.\n'
if log_file:
    with open(log_file,'a') as fd:
        fd.write(mstr)
else:
    print(mstr)
