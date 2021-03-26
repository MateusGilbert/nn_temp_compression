# Temperature Compression Using Neural Networks
This repository contains the program used to simulate the temporal compression scenarios discussed in "A Neural Network Based Asymmetrical Compression System for IoT Networks" written by Mateus S. Gilbert and Miguel Elias M. Campista. From parameters set on t2_taxonomy.py, the program trains the listed autoencoder networks to perform the compression task at hand.

The code present here was mainly written by myself, with the following exceptions:

  (1) lr_finder_keras.py, available at https://github.com/WittmannF/LRFinder;
  
  (2) clr_callback.py, available at https://github.com/bckenstler/CLR.

Also, the files containing the temperature measurements used in the experiment were generated from a dataset available at https://datadryad.org/stash/dataset/doi:10.6071/M39Q2V.

>> Obs.: Given that portuguese is my mother language, the code may contain some forgotten comments in this language that can be ignored.

# How to Use
As discussed above, the program trains a list of autoencoder networks accordingly with a set of parameters present in t2_taxonomy.py. They are the following:
