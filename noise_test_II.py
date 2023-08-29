#! /usr/bin/python3

import os
from pandas import DataFrame
import errno
import datetime as dttm
from dataHandling import get_samples
from collections import defaultdict
from my_datahanddlers import pre_process, pos_process
from tensorflow.keras.models import load_model
from testNet import ae_test_dts, enc_test_dts
from writter import wTable
from tensorflow import expand_dims
import re
import tensorflow as tf
from my_layers import MY_LAYERS

#Sample/Compression##########################
smp_size = 100
cmp_size = 25
#AE specifications###########################
out_func = 'sigmoid'#'tanh'
hid_act = 'selu'#'relu'
tempAE = 'tmpAE.h5'
bestAE = 'bestAE.h5'
worstAE = 'worstAE.h5'
randomAE = 'randomAE.h5'
models = [randomAE]#[bestAE, randomAE, worstAE]
#General#####################################
test_strides = [int(smp_size//2), int(smp_size // 3)]
func_range = (0,1)
perc_th = .01
round_range = (perc_th,func_range[1]*(1.-perc_th))
test_stride = smp_size
test_size = None
dts_name = ['YandHalfChallenge/' + name for name in ['Caples_Lake_N7_2014_20162.csv', 'Caples_Lake_N7_2016_2017.csv']] #'Caples_Lake_N7_2014_2017.csv'
cols = [0,1]
data_path = '/home/gta/gilbert/TF/YandHalfChallenge/'
if isinstance(dts_name,list):
    dts_loc = ['/home/gta/gilbert/TF/' + name for name in dts_name]
    rFolder = '{}_results'.format(dts_name[0][:-15])
else:
    dts_loc = '/home/gta/gilbert/TF/' + dts_name
    rFolder = '{}_results'.format(dts_name[:-15])
#############################################

def noise_test(models, dir_id='noise_results', noise_par=0, verbose=False, enc_type=None, gpu='/GPU:0'):
#   try:
#       os.makedirs(rFolder)
#   except OSError as err:
#       if err.errno != errno.EEXIST:
#           raise

#   os.chdir(rFolder)
    og_dir = os.getcwd()
    saveat = os.path.join(og_dir,dir_id)# + dttm.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    try:
        os.mkdir(saveat)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise
    #os.chdir(saveat)

    #generate datasets -- specific to my dataset
    if noise_par:
        test_dts = get_samples(
                                dts_loc[1],
                                smp_size,
                                strides=test_strides,
                                p_apply=1.,
                                snrs=[noise_par]
                              )
    else:
        test_dts = get_samples(
                                dts_loc[1],
                                smp_size,
                                strides=test_strides,
                              )


    if verbose:
        with open('noise_log.txt', 'a') as fd:
            fd.write(f'noise (snr): {noise_par} dB\n')
    results = defaultdict(list)
    if isinstance(models, str):
        models = models.split(' ')
    for model in models:
        if verbose:
            with open('noise_log.txt', 'a') as fd:
                fd.write(f'testing {model}\n')
        net_id = '_'.join(model.split('/')[-2:])
        #metrics initialization
        results['net_id'].append(net_id)
        results['SNR'].append(noise_par)

        #for m_type in models:
        #load best weight configuration
        if enc_type:
            path = '/'.join(model.split('/')[:-1])
            mod = re.sub('.h5', '', model.split('/')[-1])
            if enc_type == 'sparse':
                encoder = os.path.join(path,f'{mod}_enc_sparse.tflite')
            elif enc_type == 'quant':
                encoder = os.path.join(path,f'{mod}_quant_enc.tflite')
            elif enc_type == 'q-sparse':
                encoder = os.path.join(path,f'{mod}_quant_enc_sparse.tflite')
            else:
                encoder = os.path.join(path,f'{mod}_enc.tflite')
            try:
                decoder = load_model(os.path.join(path, f'{mod}_dec.h5'))
            except:
                decoder = load_model(os.path.join(path, f'{mod}_dec.h5'), custom_objects={'expand_dims': expand_dims})
            #test
            mse_err, mae_err = enc_test_dts(encoder,decoder,test_dts,verbose=verbose,encode=pre_process,decode=pos_process)
            del decoder
        else:
            try:
                ae = load_model(model)
            except:
                ae = load_model(model, custom_objects={'expand_dims': expand_dims} | MY_LAYERS)

            #test
            #with tf.device(gpu):
            if re.search('^c?PR',net_id):
                mse_err, mae_err = ae_test_dts(ae,dataset,verbose=verbose,encode=pre_process,decode=pos_process,buff_size=smp_size)
            else:
                mse_err, mae_err = ae_test_dts(ae,test_dts,verbose=verbose,encode=pre_process,decode=pos_process)
            del ae
        results['mse_err'].append(mse_err)
        results['mae_err'].append(mae_err)
        if verbose:
            with open('noise_log.txt', 'a') as fd:
                fd.write(f'finished testing {model}\n')

    df = DataFrame(results)
    saveat = os.path.join(saveat, 'noise_res.csv')
    df.to_csv(saveat, mode='a',
              header=not os.path.exists(saveat),
              index=False, float_format='%.5E')
    del df

    if verbose:
        with open('noise_log.txt', 'a') as fd:
            fd.write('----------------------------------------------------------\n\n')

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained AEs.')
    parser.add_argument('-m', '--models', nargs='+', default=[])
    parser.add_argument('-s', '--saveat', default='noise_results')
    parser.add_argument('-n', '--noise', default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-e', '--enc_type', action=None)
    parser.add_argument('-g', '--gpu', type=str, default='/GPU:0')
    args = parser.parse_args()

    models = args.models
    noise = float(args.noise)
    verbose = args.verbose
    enc_type = args.enc_type
    saveat = args.saveat
    gpu = args.gpu
    noise_test(models, dir_id=saveat, noise_par=noise, verbose=verbose, enc_type=enc_type, gpu=gpu)
