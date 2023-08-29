#! /usr/bin/python3

import os
import errno
from shutil import move as mv
import datetime as dttm
from testImports import *
from training import aeTrain, dnetTrain
from tensorflow.keras.activations import relu, sigmoid, tanh
from timeSerDataAug import jittering, flipping, scaling
from testNet import ae_test_dts#, dec_test_dts
from my_datahanddlers import pre_process, pos_process
from model_wrappers import my_wp
import tensorflow as tf
from my_layers import MY_LAYERS
import re

###trocar a loss pela huber -- vai minimizar o efeito de outliers
#Sample/Compression##########################
smp_size = 100
cmp_size = 25
#AE specifications###########################
out_func = 'sigmoid'#'tanh'
hid_act = 'selu'#'relu'
func_range = (0,1)            #talvez trocar, pq a selu não é perfeita
perc_th = .01
round_range = (perc_th,func_range[1]*(1.-perc_th))
ae_clr_const = 5    #clr stepsize constant
ae_mode = 'on_plateau'    # 'on_plateau' or 'exp_range' or  'triangular2'
ae_patience = 15    #for an ae_clr_const == 5, we have a cycle and a half
ae_fix = None#100
tempAE = 'tmpAE.h5'
bestAE = 'bestAE.h5'
worstAE = 'worstAE.h5'
randomAE = 'randomAE.h5'
#Models to be trained####################
MODELS = [
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
   ('NCAAE-1.1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
                  ('rs', (cmp_size, 1), None), ('up_conv', (4, 3, 2, 4), hid_act), ('up_conv', (4, 3, 2, 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
   ('NCAAE-1.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size,1), 4, 3, 2, 4), hid_act), ('up_conv', (4, 3, 2, 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
   ('NCAAE-1.3', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 4, 3, 2, 4), hid_act), ('d_conv', ((cmp_size*2, 4), 4, 3, 2, 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
   ('NCAAE-2.1', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 4, 3, 2, 4), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [16, 4, 4]), hid_act), ('up_conv', (8, 3, 2, 4), hid_act),
                  ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
   ('NCAAE-2.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
                  ('rs', (cmp_size, 1), None), ('d_conv', ((cmp_size, 1), 4, 3, 2, 4), hid_act), ('h_conv2', ([1, 2, 3], [4, 2, 2], 3, [16, 4, 4]), hid_act), ('d_conv', ((cmp_size*2,8), 8, 3, 2, 4), hid_act),
                  ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
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
#    ('CAAE-3.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('cv', (4, 3, 1, 'same'), hid_act), ('ct', (4, 3, 2, 'same', 4), hid_act), ('cv', (1, 3, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-4', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('cv', (4, 3, 1, 'same'), hid_act), ('ct', (16, 3, 2, 'same', 4), hid_act), ('cv', (1, 1, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
#    ('CAAE-4.2', [('in', (smp_size,), None), ('dl', cmp_size, hid_act),
#                ('dl', 50, hid_act),
#                ('rs', (50,1), None), ('cv', (4, 3, 1, 'same'), hid_act), ('ct', (16, 3, 2, 'same', 4), hid_act), ('cv', (1, 3, 1, 'same'), out_func), ('rs', (smp_size,), None)]),
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
train_strides = [10, 17, 23]
#snrs = 0 #[45, 40, 35, 30, 25, 20, 15] #in dbs
#p_apply = 2/3
test_strides = [50, 33]#smp_size
turns_per_config = 10
batch_size = 50
split_ratio = .2
usingNest = True
compType = 'temporal'
dts_name = ['YandHalfChallenge/' + name for name in ['Caples_Lake_N7_2014_20162.csv', 'Caples_Lake_N7_2016_2017.csv']] #'Caples_Lake_N7_2014_2017.csv'
#cols = [0,1]
data_path = '/home/gta/gilbert/TF/YandHalfChallenge/'
if isinstance(dts_name,list):
    dts_loc = ['/home/gta/gilbert/TF/' + name for name in dts_name]
    rFolder = '{}_results'.format(dts_name[0][:-15])
else:
    dts_loc = '/home/gta/gilbert/TF/' + dts_name
    rFolder = '{}_results'.format(dts_name[:-15])
table_name = 'results.txt'
signals_file = 'sig_file.txt'
conf_table = '{}_helper.txt'.format(table_name[:-4])
#the file above keeps of the entries in table.txt
filename = 'readme.txt'
#Configuration settings for LR finder
start_lr = 7.5e-4###################
end_lr = 1####################
lr_epochs = 15
#for a better vizualization of the plots
zoom = True
z_range = 1000
#Data Augmentation
jitt = None#([.25,.5], [.6,.4],.333)
flip = None#.005
scal = None#.125
combine = None
#############################################

def t2_taxonomy(models=MODELS,snrs=0,p_apply=0.,lr_list=None,dir_id='tx_results_',cols=None,verbose=False,log_file=None, gpu='/GPU:0'):
    try:
        os.makedirs(rFolder)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise

    os.chdir(rFolder)
    execDirname = dir_id + dttm.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    os.mkdir(execDirname)
    os.chdir(execDirname)

    #generate datasets -- specific to my dataset
#   try:
#       idxs = list()
#       for name in ['train', 'val']:
#           with open(os.path.join(data_path, f'{name}_idx_Caples_Lake_N7.txt'), 'r') as f:
#               idxs.append(np.loadtxt(f,dtype=int))
#   except:
#       idxs = None

#   if idxs:
#       train_idxs,val_idxs = idxs
#       train_dts = get_samples(
#                               dts_loc[0],
#                               smp_size,
#                               strides=train_strides,
#                               snrs=snrs,
#                               p_apply=p_apply,
#                               samples=train_idxs,
#                               encode=pre_process,
#                               B_size=batch_size,
#                           )
#       val_dts = get_samples(
#                               dts_loc[0],
#                               smp_size,
#                               strides=train_strides,
#                               snrs=snrs,
#                               p_apply=p_apply,
#                               samples=val_idxs,
#                               encode=pre_process,
#                               B_size=batch_size,
#                           )
#   else:
#       train_val_data = get_samples(
#                               dts_loc[0],
#                               smp_size,
#                               strides=train_strides,
#                               snrs=snrs,
#                               p_apply=p_apply,
#                               encode=pre_process,
#                               B_size=batch_size,
#                           )

#       train_idxs,val_idxs = train_val_data.get_indices(split_ratio)
#       for name, idxs in zip(('train', 'val'), (train_idxs,val_idxs)):
#           with open(os.path.join(data_path, f'{name}_idx_Caples_Lake_N7.txt'), 'w') as f:
#               np.savetxt(f, idxs, fmt='%d')

#       train_dts = train_val_data.subset(train_idxs)
#       val_dts = train_val_data.subset(val_idxs)

    train_dts = get_samples(
                            dts_loc[0],
                            smp_size,
                            strides=train_strides,
                            snrs=snrs,
                            p_apply=p_apply,
                            encode=pre_process,
                            #B_size=batch_size,
                          )
    test_dts = get_samples(
                            dts_loc[1],
                            smp_size,
                            strides=test_strides,
                          )

    aug_tec = []
    #scaling
    if scal:
        train_dts,_ = scaling(train_dts,scale_y=True,addProb=scal)
        aug_tec.append('Scaling ({:.3f})'.format(scal))
    #flipping
    if flip:
        train_dts,_ = flipping(train_dts,addProb=flip)
        aug_tec.append('Flipping ({:.3f})'.format(flip))
    #jittering
    if jitt:
        train_dts,_ = jittering(train_dts, sigma=jitt[0],
            prob=jitt[1], addProb=jitt[2])
        aug_tec.append('Jittering ({:.3f}) with σ = {} with probabilities {}, respectively'.format(
            jitt[2],jitt[1], jitt[0]))
    if not len(aug_tec):
        aug_tec = ['None']

    if usingNest:
        optFunc = Nadam
    else:
        optFunc = Adam

    results_dir = os.getcwd()
    with open(conf_table, 'w') as fd:
        fd.write('This file keeps track of each AE  configuration.\n')
        fd.write('Each line presents the number of neurons per layers.\n')
        fd.write('Each line is correspond to each line of {}, in order.\n'.format(table_name))
        fd.write('-'*20 + '\n')
        fd.write('From {} down to {}\n'.format(smp_size,cmp_size))
        fd.write('-'*20 + '\n')

    for net_id,model in models:
        with open(conf_table, 'a') as fd:
            fd.write(f'{net_id} (model list): {model}\n')
        config_dirname = '_'.join(net_id.split('-'))
        os.makedirs(config_dirname)
        os.chdir(config_dirname)

        mstr = f'Training Config.: {net_id}\n'
        if log_file:
            with open(log_file,'a') as fd:
                fd.write(mstr)
        else:
            print(mstr)

        if lr_list and net_id in lr_list.keys():
            (min_lr,max_lr) = lr_list[net_id]
        elif ae_mode == 'on_plateau':
#           if snrs:
#               min_lr,max_lr = 5e-5,1e-3
#           else:
            min_lr,max_lr = 5e-5,3e-3
        else:
            ae = buildNN(model)
            #ae = my_wp(model[0][1], ae)#, pre_process, pos_process)#######modificacao
            ae.compile(loss=MeanSquaredError(), optimizer=optFunc(), metrics=['mse','mae'])

            lr_finder = LRFinder(min_lr=start_lr, max_lr=end_lr)
            ae.fit(inputs,targets,batch_size=batch_size,callbacks=[lr_finder],epochs=lr_epochs)
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
            del ae

        #metrics initialization
        best_outputs = []
        worst_outputs = []
        random_outputs = []
        best_loss = None; worst_loss = None; random_loss = None
        for i in range(turns_per_config):
            res = {'net_id': [net_id], 'mean_mse': [np.NaN],
                    'mean_mae': [np.NaN], 'its': [np.NaN]}
            time_str = dttm.datetime.now().strftime('%H:%M:%S %m/%d')
            mstr = f'>>>{i:03d} Turn\tStarted at {time_str}\n'
            if log_file:
                with open(log_file,'a') as fd:
                    fd.write(mstr)
            else:
                print(mstr)
            with tf.device(gpu):
                iterations = aeTrain(model,#layers.copy(),
                            batch_size,
                            train_dts,
                            (min_lr,max_lr,ae_mode),
                            k=ae_clr_const,
                            patience=ae_patience,
                            saveAt=tempAE,
                            verbose=verbose,
                            wait=ae_fix,
                            optimizer=optFunc,
                            val_size=split_ratio)
                            #val_set=val_dts)
    #                        regs=(lays,reg))
            res['its'][0] = iterations

            end_str = dttm.datetime.now().strftime('%H:%M:%S %m/%d')
            mstr = f'It took {iterations} iterations ({end_str})\n'
            if log_file:
                with open(log_file,'a') as fd:
                    fd.write(mstr)
            else:
                print(mstr)
            #load best weight configuration
            try:
                ae = load_model(tempAE)
            except ValueError:
                ae = load_model(tempAE, custom_objects=MY_LAYERS)#{'OrthogonalRegularizer': OrthogonalRegularizer})
            os.remove(tempAE)

            #test
            mse_err, mae_err = ae_test_dts(ae,test_dts,verbose=verbose, encode=pre_process, decode=pos_process)
            res['mean_mse'][0] = mse_err
            res['mean_mae'][0] = mae_err
            #cur_outputs = aux_outputs
            curTestLoss = mse_err

            ##save results and free memory
            df = DataFrame(res)
            df.to_csv(table_name, mode='a',
                    header=not os.path.exists(table_name),
                    index=False, float_format='%.5E')
            del df
            ae.save(f'model_{i+1}.h5')

            #update results
#           if len(best_outputs):
#               if minLoss > curTestLoss:
#                   minLoss = curTestLoss
#                   ae.save(bestAE)
#               elif maxLoss < curTestLoss:
#                   maxLoss = curTestLoss
#                   ae.save(worstAE)
#               elif not np.random.randint(0,100) % 5:            #.2 prob of selecting results
#                   ae.save(randomAE)
#           else:
#               minLoss = maxLoss = curTestLoss
#               ae.save(bestAE); cp(bestAE,worstAE); cp(bestAE,randomAE)
            del ae

        os.chdir('..')

    return results_dir

if __name__ == '__main__':
    print('Checking if models are viable')
    print('There are {} models'.format(len(models)))
    for net_id,model in models:
        print('\nNet id {}'.format(net_id))
        print(model)
    #    try:
        ae = buildNN(model)
        #ae = my_wp(model[0][1], ae)#, pre_process, pos_process)#######modificacao
    #    except:
    #        for i in range(1,len(model)+1):
    #            ae = buildNN(model[:i])
    #            ae.summary()
    #    if net_id in reg_dict.keys():
    #        layers, reg = reg_dict[net_id]
    #        add_lregulirizer(ae, reg, layers)
        ae.summary()
        del ae
    print('Success!!!')
