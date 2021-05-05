# Temperature Compression Using Neural Networks
This repository contains the program used to simulate the temporal compression scenarios discussed in "A Neural Network Based Asymmetrical Compression System for IoT Networks" written by Mateus S. Gilbert and Miguel Elias M. Campista. From parameters defined in t2_taxonomy.py, the program trains the listed autoencoder networks to perform the compression task at hand.

The code present here was mainly written by myself, with the following exceptions:

  (1) lr_finder_keras.py, available at https://github.com/WittmannF/LRFinder;
  
  (2) clr_callback.py, available at https://github.com/bckenstler/CLR.

Also, the files containing the temperature measurements used in the experiment were extracted from a dataset available at https://datadryad.org/stash/dataset/doi:10.6071/M39Q2V.

>> Obs.: Given that portuguese is my mother language, the code may contain some forgotten comments in this language that can be ignored.
>> Obs2.: Documentation still in progress. The information presented below is sufficient to understand the requirements for running the program. Feel free to contact me if you have any questions about the files not discussed below.

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

In the same file, the neural networks to be evaluated are kept in a list structure, as demonstrated below:

![test image size](https://github.com/MateusGilbert/nn_temp_compression/blob/main/pics/ae_list.png)

Each network is defined by a string, which will be used to identify its results. The network architecture is maintained on its own list, which will be the input variable of a program that actually constructs the neural network (to be discussed later). Both values are grouped in a pair, as can be seen in the picture. For instance, the pair ('AE-3', [('in',(smp_size,),None), ('dl',65,hid_act), ('dl',cmp_size,hid_act), ('dl',65,hid_act), ('dl',smp_size,out_func)]) represents a symmetrical autoencoder (labeled as AE-3) with a smp_size-65-cmp_size-65-smp_size architecure (if smp_size=100 and cmp_size=25, it would be 100-65-25-65-100), where all except the output layer have hid_act as its activation function. Again, more details will be discussed when the program that actually constructs the neural networks is presented.

Lastly, the following parameters are needed to start the program:

![test image size](https://github.com/MateusGilbert/nn_temp_compression/blob/main/pics/gen_par.png)

(1) wnd_stride defines the stride rate used in the training set to generate training batches;

(2) wnd_stride_comp keep the aditional stride rates to be used in the training set, if desired (if not, should be set to None);

(3) test_stride defines the stride rate to be used in the test set (if no overlap is desired, should be set to >= smp_size);

(4) turns_per_config defines the number of times each neural network model will be trained (and tested);

(5) batch_size defines the number of samples that will form a training batch;

(6) test_size defines the percentage of samples from the desired dataset that will be spared for testing;

(7) usingNest defines if the Nadam should be used in trainig (when set to false, Adam is the selected optimizer);

(8) compType defines if the scenario analysed is temporal or spatial (obs.: the later has not yet been implemented);

(9) dts_name defines the datasets to be used;

(10) cols defines the columns to be extracted from the dataset;

(11) table_name defines the file where the results will be save;

(12) conf_table is the file that will list the order of the models shown by the file that keeps the results (it is an auxiliary file to the previous one);

(13) filename is the readme file that is generated at the end of the program;

(14) (start|end)_lr and lr_epochs are parameters needed to run LRFinder;

(15) when zoom is set, generates subplots of the test results taking z_range entries;

(16) variables that fall under '#Data Augmentation' set the parameters of these augmentation algorithms (to be discussed later).
