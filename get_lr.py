#! /usr/bin/python3.6

import os
import errno
from shutil import move as mv
import datetime as dttm
from testImports import *
from training import aeTrain, dnetTrain
from tensorflow.keras.activations import relu, sigmoid, tanh
from timeSerDataAug import jittering, flipping, scaling, jittering_awgn
from testNet import ae_test_dts#, dec_test_dts
from glob import glob
import tensorflow as tf
from numpy.random import seed
from my_datahanddlers import pre_process, pos_process
from model_wrappers import my_wp
from numpy import concatenate, newaxis as nax

###trocar a loss pela huber -- vai minimizar o efeito de outliers
#Sample/Compression##########################
smp_size = 100#25      #5s
cmp_size = 25#50
#AE specifications###########################
out_func = 'sigmoid'
hid_act = 'selu'#'relu'
hid_act2 = 'relu'
enc_act = None#'c_sigmoid'
func_range = (0,1)            #talvez trocar, pq a selu não é perfeita
perc_th = .05
round_range = (perc_th,func_range[1]*(1.-perc_th))
ae_clr_const = 5    #clr stepsize constant
ae_mode = 'exp_range'    #'triangular2'
ae_patience = 15    #for an ae_clr_const == 5, we have a cycle and a half
ae_fix = None#100
tempAE = 'tmpAE.h5'
bestAE = 'bestAE.h5'
worstAE = 'worstAE.h5'
randomAE = 'randomAE.h5'
#Models to be trained####################
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
#    ('AAE-1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act), ('dl', 50, hid_act), ('dl', smp_size, out_func)]),
#    ('AAE-2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act), ('dl', 75, hid_act), ('dl', smp_size, out_func)]),
#    ('AAE-3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 45, hid_act), ('dl', 65, hid_act), ('dl', 85, hid_act), ('dl', smp_size, out_func)]),
#    ('AAE-4', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 40, hid_act), ('dl', 55, hid_act), ('dl', 70, hid_act), ('dl', 85, hid_act), ('dl', smp_size, out_func)]),

    #CAAEs
#   ('NCAAE-1.1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                  ('rs', (cmp_size, 1), None), ('up_conv', (4, 3, 2, 4), hid_act), ('up_conv', (4, 3, 2, 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
   ('NCAAE-1.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size,1), 4, 3, 2, 4), hid_act), ('up_conv', (4, 3, 2, 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#   ('NCAAE-1.3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 4, 3, 2, 4), hid_act), ('d_conv', ((cmp_size*2, 4), 4, 3, 2, 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#   ('NCAAE-2.1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 4, 3, 2, 4), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [16, 4, 4]), hid_act), ('up_conv', (8, 3, 2, 4), hid_act),
#                  ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#   ('NCAAE-2.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 4, 3, 2, 4), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [16, 4, 4]), hid_act), ('d_conv', ((cmp_size*2,8), 8, 3, 2, 4), hid_act),
#                  ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act), ('rs', (50,1), None), ('ct', (4, 3, 2, 'same'), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-1.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act), ('rs', (50,1), None), ('ct', (4, 3, 2, 'same'), hid_act), ('cv', (1, 3, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('cv', (4, 3, 1, 'same'), hid_act), ('ct', (4, 3, 2, 'same', 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-2.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('cv', (4, 3, 1, 'same'), hid_act), ('ct', (4, 3, 2, 'same', 4), hid_act), ('cv', (1, 3, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('cv', (4, 3, 1, 'same'), hid_act), ('ct', (4, 3, 2, 'same', 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('h_cv', ([1,2,3,4], 3, 1, 'same'), hid_act), ('ct', (4, 3, 2, 'same', 4), hid_act), ('cv', (1, 3, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-4', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('cv', (4, 3, 1, 'same'), hid_act), ('ct', (16, 3, 2, 'same', 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-4', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('h_cv', ([1,2,3,4], 3, 1, 'same'), hid_act), ('ct', (16, 3, 2, 'same', 4), hid_act), ('cv', (1, 3, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
]
# Regularizers dict######################
reg_dict = {
        # model, layers, regularizers
        #Teste 1
#        'AAE-1r1': ([1, 1], [('kernel_regularizer', 'rort', .1), ('activity_regularizer', 'l1', .025)]),
#        'AAE-1r2': ([1, 1], [('kernel_regularizer', 'rort', .25), ('activity_regularizer', 'l1', .025)]),
#        'AAE-1r3': ([1, 1], [('kernel_regularizer', 'rort', .5), ('activity_regularizer', 'l1', .025)]),
#
#        'AAE-2r1': ([1, 1], [('kernel_regularizer', 'rort', .1), ('activity_regularizer', 'l1', .025)]),
#        'AAE-2r2': ([1, 1], [('kernel_regularizer', 'rort', .25), ('activity_regularizer', 'l1', .025)]),
#        'AAE-2r3': ([1, 1], [('kernel_regularizer', 'rort', .5), ('activity_regularizer', 'l1', .025)]),
#
#        'AAE-3r1': ([1, 1], [('kernel_regularizer', 'rort', .1), ('activity_regularizer', 'l1', .025)]),
#        'AAE-3r2': ([1, 1], [('kernel_regularizer', 'rort', .25), ('activity_regularizer', 'l1', .025)]),
#        'AAE-3r3': ([1, 1], [('kernel_regularizer', 'rort', .5), ('activity_regularizer', 'l1', .025)]),
#
#        'AAE-4r1': ([1, 1], [('kernel_regularizer', 'rort', .1), ('activity_regularizer', 'l1', .025)]),
#        'AAE-4r2': ([1, 1], [('kernel_regularizer', 'rort', .25), ('activity_regularizer', 'l1', .025)]),
#        'AAE-4r3': ([1, 1], [('kernel_regularizer', 'rort', .5), ('activity_regularizer', 'l1', .025)]),

        #Teste 2
#        'CAAE-1r1': ([1, 4], [('kernel_regularizer', 'rort', .05), ('activity_regularizer', 'cort', .02)]),
#        'CAAE-1r2': ([1, 4], [('kernel_regularizer', 'rort', .05), ('activity_regularizer', 'cort', .05)]),
#        'CAAE-1r5': ([1, 4, 4], [('kernel_regularizer', 'rort', .01), ('kernel_regularizer', 'l2', .02), ('activity_regularizer', 'cort', .01)]),
#        'CAAE-1r6': ([1, 4, 4], [('kernel_regularizer', 'rort', .01), ('kernel_regularizer', 'l2', .05), ('activity_regularizer', 'cort', .01)]),

#        'CAAE-2r1': ([1, [14,15,16,17]], [('kernel_regularizer', 'rort', .01), ('activity_regularizer', 'cort', .01)]),
#        'CAAE-2r2': ([1, [14,15,16,17]], [('kernel_regularizer', 'rort', .005), ('activity_regularizer', 'cort', .005)]),
#        'CAAE-2r3': ([1, [14,15,16,17], [14,15,16,17]], [('kernel_regularizer', 'rort', .01), ('kernel_regularizer', 'l2', .1), ('activity_regularizer', 'cort', .01)]),
#        'CAAE-2r4': ([1, [14,15,16,17], [14,15,16,17]], [('kernel_regularizer', 'rort', .005), ('kernel_regularizer', 'l2', .1), ('activity_regularizer', 'cort', .005)]),
#        'CAAE-2r5': ([1, [14,15,16,17], [14,15,16,17]], [('kernel_regularizer', 'rort', .01), ('kernel_regularizer', 'l2', .2), ('activity_regularizer', 'cort', .01)]),
#        'CAAE-2r6': ([1, [14,15,16,17], [14,15,16,17]], [('kernel_regularizer', 'rort', .005), ('kernel_regularizer', 'l2', .2), ('activity_regularizer', 'cort', .005)]),
}

#General#####################################
#wnd_stride = 10# 4s
train_strides = [20]
snrs = [20, 10, 5] #in dbs
p_apply = 2/5
#wnd_str_comp = None#[63] #~2.5s
#test_stride = smp_size
turns_per_config = 10
batch_size = 20
test_size = .2
usingNest = True
compType = 'temporal'
dts_name = ['YandHalfChallenge/' + name for name in ['Caples_Lake_N7_2014_20162.csv', 'Caples_Lake_N7_2016_2017.csv']] #'Caples_Lake_N7_2014_2017.csv'
#dts_loc = '/home/mateusgilbert/trabalhos/nn_temp_compression/Datasets/'#data'
if isinstance(dts_name,list):
	dts_loc = ['./Datasets/' + name for name in dts_name]
	rFolder = '{}_results'.format(dts_name[0][:-6])
else:
	dts_loc = './Datasets/' + dts_name
	rFolder = '{}_results'.format(dts_name[:-6])
rFolder = f'{dts_name}_results'
setup_file = f'{dts_loc}/aae_class.txt'
#tr_indices = dts_loc + '/train_ids.txt'
#te_indices = dts_loc + '/test_ids.txt'
#table_name = 'table.csv'
#conf_table = '{}_helper.txt'.format(table_name[:-4])
#the file above keeps of the entries in table.txt
#filename = 'readme.txt'
#Configuration settings for LR finder
start_lr = 8e-5###################
end_lr = 5e-2####################
lr_epochs = 20
#for a better vizualization of the plots
zoom = True
z_range = 1000
#Data Augmentation
jitt = None#([15., 7.5], 1.)
flip = None#.005
scal = None#(.1,.1)
combine = True
verbose = True
#############################################

#####
seed(1671)
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#tf.keras.backend.tensorflow_backend.set_session(sess)
tf.compat.v1.keras.backend.set_session(sess)
#get_custom_objects().update({'custom_sigmoid': custom_sigmoid})
#####


#encode law for classification problem
def encode_law(directory,use_map):
    n = use_map['n']
    return [1 if i == use_map[a_id]  else 0 for i in range(n)] 
#    a_id = int(directory[1:].split('/')[0]) - 1 #19 activities
#    return [1 if i == a_id  else 0 for i in range(19)] 

enc_law = encode_law

#   setup_file -- definir test_list; modificar a tabela (colocar tds as informações + separar add test)
#with open(setup_file,'r') as fd:
#    act_list = next(fd)[:-1].split(',')
#    if enc_law:
#        use_map = {'n':len(act_list)}
#        for i,act in enumerate(act_list):
#            use_map[act] = i
#    else:
#        use_map = None
#    sig_file = next(fd)[:-1]
#    aux = next(fd)
#    people_list = aux[:-1].split(',')
#    dts_list = list()
#    for act in act_list: 
#        for person in people_list:
#            dts_list.append(os.path.join(act,person,sig_file))
#    dts_list.sort()
#    
#    aux = next(fd)
#    test_list = None
#    if aux:
#        test_list = list()
#        people_list = aux[:-1].split(',')
#        for act in act_list: 
#            for person in people_list:
#                dirname = os.path.join(dts_loc,act,person)
#                files = [i.split('/')[-1] for i in glob(os.path.join(dirname,'T*.csv'))]
#                for sig_file in files:
#                    test_list.append(os.path.join(act,person,sig_file))
#    del aux
cols = [0,1]# [f'T_{i}acc' for i in ['x', 'y', 'z']] 
train_dts = get_samples(
                        dts_loc[0],
                        smp_size,
                        strides=train_strides,
                        snrs=snrs,
                        p_apply=p_apply,
                        encode=pre_process,
                        #B_size=batch_size,
                      )
#generate datasets -- specific to my dataset
#(train_dts,num_batches),(test_dts,num_ts_batches) = initDataset(dts_loc,
#                                cols,
#                                smp_size,
#                                test_size,
#                                wnd_stride,
#                                test_stride,
#                                wnd_str_comp=wnd_str_comp)#,#combine=combine)
#(train_dts,num_batches),(test_dts,add_dts,num_ts_batches) = initDataset(dts_list,
#                                cols,
#                                smp_size,
#                                test_size,
#                                wnd_stride,
#                                test_stride,
#                                wnd_str_comp=wnd_str_comp,
#                                encode_law=enc_law,
#                                use_map=use_map,#####################
#                                combine_cols=combine,
#                                test_list=test_list,
#                                path=dts_loc)
#    (train_dts,num_batches),(test_dts,num_ts_batches) = initDataset_files(tr_indices,
#                                    te_indices,
#                                    encode_law=enc_law,
#                                    columns=cols,
#                                    combine=combine)

aug_tec = []
#scaling
if scal:
    train_dts,num_batches = scaling(train_dts,sigma=scal[0],scale_y=True,addProb=scal[1])
    aug_tec.append('Scaling ({:.3f})'.format(scal[1]))
#flipping
if flip:
    train_dts,num_batches = flipping(train_dts,addProb=flip)
    aug_tec.append('Flipping ({:.3f})'.format(flip))
#jittering
if jitt:
    if isinstance(jitt, tuple):
        train_dts,num_batches = jittering_awgn(train_dts, sigma=jitt[0],
            addProb=jitt[1], prob=1)
        aug_tec.append('Jittering ({:.3f}) with σ_dB = {} with probabilities {}, respectively'.format(
            jitt[1],jitt[0],None))
    else:
        train_dts,num_batches = jittering_awgn(train_dts, sigma=jitt[0],
            prob=jitt[1], addProb=jitt[2])
        aug_tec.append('Jittering ({:.3f}) with σ_dB = {} with probabilities {}, respectively'.format(
            jitt[2],jitt[1], jitt[0]))
#        aug_tec.append('Jittering ({:.3f}) with σ = {} with probabilities {}, respectively'.format(
#            jitt[2],jitt[1], jitt[0]))
if not len(aug_tec):
    aug_tec = ['None']

#unscalled_dts = train_dts
##scale each training batch inside function output range and shuffle them
#if enc_law:
#    train_dts = scaleBatches(
#        train_dts,dRange=round_range,s_ipts=True,s_tgs=False).shuffle(buffer_size=num_batches)
#else:
#    train_dts = scaleBatches(
#        train_dts,dRange=round_range,s_ipts=True,s_tgs=True).shuffle(buffer_size=num_batches)
#del unscalled_dts
#
#inputs = []
#targets = []
##transpose vectors to use as input for the network
#for x,y in train_dts:
#    if x.ndim == 1:
#        inputs.append(x.numpy()[nax,:])
#    elif x.shape[1] == 1:
#        inputs.append(x.numpy().T)
#    else:
#        inputs.append(x.numpy())
#    if y.ndim == 1:
#        targets.append(y.numpy()[nax,:])
#    elif y.shape[1] == 1:
#        targets.append(y.numpy().T)
#    else:
#        targets.append(y.numpy())
#inputs = concat(inputs,0)
#targets = concat(targets,0)

#unscalled_dts = train_dts
##scale each training batch inside function output range and shuffle them
#train_dts = scaleBatches(
#    train_dts,dRange=round_range,s_ipts=True,s_tgs=True).shuffle(buffer_size=num_batches)

inputs = []
targets = []
for x,y in train_dts:
    inputs.append(x.numpy()[nax,:])
    targets.append(y.numpy()[nax,:])
inputs = concatenate(inputs,axis=0)
targets = concatenate(targets,axis=0)
inputs = convert_to_tensor(inputs)
targets = convert_to_tensor(targets)

if usingNest:
    optFunc = Nadam
else:
    optFunc = Adam

lr_dict = {'net_id': list(), 'max_lr': list(), 'min_lr': list()}
for net_id,model in models:
    lr_dict['net_id'].append(net_id)

#    #define ae
#    if net_id in reg_dict.keys():
#        lays, reg = reg_dict[net_id]
#    else:
#        lays,reg = None,None
#
#    ae = buildNN(model)
#    if reg:
#        add_lregulirizer(ae, reg, lays)
#    ae.compile(loss=MeanSquaredError(), optimizer=optFunc(), metrics=['mse','mae'])

    if isinstance(model[0], list):
        if len(model) == 2:
            d_net = multInputsNN(model[0], model[1])
        else:
            inp, brns, out = model
            d_net, n_inp = multOutputsNN(inp, brns, return_model=False)
            d_net = buildNN(out, prev_layers=d_net, return_model=False)[0]
            d_net = Model(inputs=n_inp, outputs=ae)
    else:
        d_net = buildNN(model)
        #d_net = my_wp(model[0][1], d_net, pre_process, pos_process)#######modificacao

#    if regs[0]:
#        lays, reg = regs
#        add_lregulirizer(ae, reg, lays)

#    if base_ae:
#        #praae_init(ae,base_ae)
#        ex_layers = kargs['ex_layers']
#        copy_to = kargs['copy_to'] if 'copy_to' in kargs.keys() else None
#        get_pretrained(d_net,base_ae,ex_layers,copy_to=copy_to)
#        #add diff learning rate
#        optimizer = MultiOptimizer([(optimizer(learning_rate=lr_vals*.05), d_net.pr_layers),
#                                    (optimizer(learning_rate=lr_vals), d_net.new_layers)])######


    if verbose:
        d_net.summary()

#    if base_ae:
#        d_net.compile(loss=SparseCategoricalCrossentropy(), optimizer=optimizer, metrics=['accuracy','f1'])
#    else:
#        d_net.compile(loss=SparseCategoricalCrossentropy(), optimizer=optimizer(), metrics=['accuracy','f1'])
    lr_finder = LRFinder(min_lr=start_lr, max_lr=end_lr)
    
    #ae.fit(inputs,targets,batch_size=batch_size,callbacks=[lr_finder],epochs=lr_epochs)
    d_net.compile(loss=MeanSquaredError(), optimizer=optFunc(), metrics=['mse','mae'])
    d_net.fit(inputs,targets,batch_size=batch_size,callbacks=[lr_finder],epochs=lr_epochs)
    min_lr = float(input('Insert minimum value for Learning Rate: '))
    max_lr = float(input('Insert maximum value for Learning Rate: '))
    while(True):
        print('min_lr = {:.2g}; max_lr = {:.2g}'.format(min_lr,max_lr))
        op = input("Want to change one of the limits? [Y]es/[N]o: ")
        if op[0].upper() == 'N':
            break
        op = int(input('\t[0] for min_lr\n\t[1] for max_lr\n\t[2]Both\nOption:'))
        if not op % 2:
            min_lr = float(input('Insert minimum value for Learning Rate: '))
            if op == 2:
                max_lr = float(input('Insert maximum value for Learning Rate: '))
        else:
            max_lr = float(input('Insert maximum value for Learning Rate: '))
    del d_net
    lr_dict['max_lr'].append(max_lr)
    lr_dict['min_lr'].append(min_lr)

df = DataFrame(lr_dict)
print(df)
op = input('Save learning rates? [y|n]')
if op.lower()[0] == 'y':
    filename = input('Insert file name: ')
    df.to_csv(filename,mode='a',
                header=not os.path.exists(filename),
                index=False, float_format='%.3E')
