#! /usr/bin/python3

import numpy as np
from testNet import ae_test_dts, dec_test_dts
from dataHandling import matrixSplit as msplit
from dataHandling import scale, stdDev, setDataset, scaleBatches, initDataset
from func import training
from neuralNets import buildNN, custom_sigmoid
from plotter import plot2D, plfromTab, barFromTab
from writter import wTable, wReadme, wReadme2
from reader import rSpecifications, searchFiles
import matplotlib.pyplot as pl
import os
from shutil import copyfile as cp
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error as MSE
#from sklearn.metrics import mean_squared_log_error as MSLE
from tensorflow import data as dtHand		#i've the custom of naming variables data
from tensorflow import transpose, convert_to_tensor
from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow import concat
######################################
from clr_callback import CyclicLR
from lr_finder_keras import LRFinder
######################################
import tensorflow.nn as nn_utils
from numpy import newaxis as nax
from genDataset import genDataset
import re
