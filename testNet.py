#! /usr/bin/python3

from numpy import newaxis as nax, zeros, argmax
import numpy as np #remover os outros, colocar np em td
from numpy.random import normal, choice
from tensorflow.keras import Sequential
from tensorflow.keras.metrics import Mean
from dataHandling import scale
from tensorflow import transpose, convert_to_tensor
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE
from tensorflow import expand_dims
import tensorflow.lite as tflite
#from class_toolbox import get_functions, eval_pred

#dps acrescentar random noise
def ae_test_dts(ae_model, dataset, verbose=True, t_noise=None, return_losses=False, encode=None, decode=None):
    outputs = list()
    mse_err = list()
    mae_err = list()

    if t_noise:
        noisy_inputs = list()
        m, s, random_burst = t_noise
        if random_burst:
            mag, prob = random_burst
    for i,(x,y) in enumerate(dataset):
        if t_noise:
            y += convert_to_tensor(normal(m,s,y.shape))
            if random_burst and choice([True, False], p=[prob,1.-prob]):
                y += convert_to_tensor(mag*noise(0,s,y.shape))
            noisy_inputs += y.numpy().squeeze().tolist()

        encoded = encode(x.numpy())
        output = ae_model(encoded[0][nax,:],training=False)#y_scaled.T,training=False)
        y_out = decode(output.numpy().squeeze(), *encoded[1:])

        y_ref = y.numpy()#.tolist()#[:test_stride]
        mse_err.append(MSE(y_true=y_ref,y_pred=y_out)) #mudei para listas
        mae_err.append(MAE(y_true=y_ref,y_pred=y_out))

        del y_ref, y_out, x, y, encoded, output

    if return_losses:
        return mse_err, mae_err
    return np.mean(mse_err), np.mean(mae_err)

def enc_test_dts(encoder, decoder, dataset, verbose=True, t_noise=None, return_losses=False, encode=None, decode=None):
    outputs = list()
    mse_err = list()
    mae_err = list()

    if t_noise:
        noisy_inputs = list()
        m, s, random_burst = t_noise
        if random_burst:
            mag, prob = random_burst

    #initialize tf_lite model
    interpreter = tflite.Interpreter(model_path=encoder)
    interpreter.allocate_tensors()
    inp_shape = interpreter.get_input_details()[0]
    out_shape = interpreter.get_output_details()[0]
    for i,(x,y) in enumerate(dataset):
        if t_noise:
            y += convert_to_tensor(normal(m,s,y.shape))
            if random_burst and choice([True, False], p=[prob,1.-prob]):
                y += convert_to_tensor(mag*noise(0,s,y.shape))
            noisy_inputs += y.numpy().squeeze().tolist()

        encoded = encode(x.numpy())
        interpreter.set_tensor(inp_shape['index'], encoded[0][nax,:])#.astype(np.float16))
        interpreter.invoke()
        cmp_vector = interpreter.get_tensor(out_shape['index']).squeeze()#.astype(np.float32)

        output = decoder(cmp_vector[nax,:],training=False)         #conferir
        y_out = decode(output.numpy().squeeze(), *encoded[1:])

        y_ref = y.numpy()#.tolist()#[:test_stride]
        mse_err.append(MSE(y_true=y_ref,y_pred=y_out)) #mudei para listas
        mae_err.append(MAE(y_true=y_ref,y_pred=y_out))

        del y_ref, y_out, x, y, encoded, output

    del interpreter
    if return_losses:
        return mse_err, mae_err
    return np.mean(mse_err), np.mean(mae_err)

def dnet_test_dts(net, dataset, func_range, n_class, verbose=True, top_k=None, it=None, save_pre=None):
    conf_matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for x,y in dataset:
        if top_k:
            if 'Y_og' in locals():
                Y_og.append(y.numpy())
            else:
                Y_og = [y.numpy()]
        (x_scaled,mins,maxs) = scale(x.numpy()[:,nax],func_range[0],func_range[1])
        #conversion to np.array of shape (1,n) because of njit 
        #functions (see dataHandling.py)
        x_scaled = x_scaled.T

        out = net(x_scaled)
        if top_k:
            if 'Y_pred' in locals():
                Y_pred.append(out.numpy())
            else:
                Y_pred = [out.numpy()]
        y_i = argmax(y.numpy())
        y_j = argmax(out.numpy())
        conf_matrix[y_i][y_j] += 1

    if verbose:
        print('confusion matrix:')
        for line in conf_matrix:
            print(line)

    #save confusion matrix
    save_at = f"conf_mat_{it or '_'}.csv"
    if save_pre:
        save_at = save_pre + save_at

    with open(save_at,'w') as c_matrix:
        for row in conf_matrix:
            for col in row:
                c_matrix.write(f'{col},')
            c_matrix.write('\n')

    #save top_k results
    functions = get_functions(['acc', 'prc', 'rec', 'f1'], top_k=top_k)
    if top_k:
        res = pd.DataFrame(eval_pred(Y_og,Y_pred,functions,add_col=('top_k',top_k))).set_index('top_k')
        if verbose:
            print(res)

        save_at = f"top_k_i{it or '_'}.csv"
        if save_pre:
            save_at = save_pre + save_at
        res.to_csv(save_at)

    return
