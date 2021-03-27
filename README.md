# Temperature Compression Using Neural Networks
This repository contains the program used to simulate the temporal compression scenarios discussed in "A Neural Network Based Asymmetrical Compression System for IoT Networks" written by Mateus S. Gilbert and Miguel Elias M. Campista. From parameters defined in t2_taxonomy.py, the program trains the listed autoencoder networks to perform the compression task at hand.

The code present here was mainly written by myself, with the following exceptions:

  (1) lr_finder_keras.py, available at https://github.com/WittmannF/LRFinder;
  
  (2) clr_callback.py, available at https://github.com/bckenstler/CLR.

Also, the files containing the temperature measurements used in the experiment were extracted from a dataset available at https://datadryad.org/stash/dataset/doi:10.6071/M39Q2V.

>> Obs.: Given that portuguese is my mother language, the code may contain some forgotten comments in this language that can be ignored.

# How to Use
As discussed above, the program trains a list of autoencoder networks accordingly with a set of parameters present in t2_taxonomy.py. They are the following:

![test image size](https://github.com/MateusGilbert/nn_temp_compression/blob/main/pics/dim_def.png)

smp_size indicates the amount of samples are taken for each compressoon, whereas cmp_size indicates the size this batch of samples will be compressed to.

![test image size](https://github.com/MateusGilbert/nn_temp_compression/blob/main/pics/ae_spec.png)

(1) out_func defines the activation function of the AEs output layer;

(2) hid_act(2) the hidden layers activation function;

(3) enc_activation defines the encoder's output layer activation function;

(4) func_range keeps the interval to which the samples will be scalled to;

(5) perc_th defines a percentage to be removed from the above interval (ex.: func_range = (0,1) and perc_th = 5e-4 ⇒ "new" func_range = (5e-4, 1 - 5e-4));

(6) ae_clr_const defines the cyclical learning rate (clr) stepsize constant (see https://github.com/bckenstler/CLR)

(7) ae_mode define clr variation policy (see https://github.com/bckenstler/CLR);

(8) ar_patience defines early stopping patience;

(9) (temp|best|worst|random)AE keep the name of the files that keep the best AE configuration during training and the best, worst and a random trained model after all the training rounds for each architecture, respectively.
