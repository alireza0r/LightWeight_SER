import torch
import torch.nn as nn
import numpy as np
from time import time
import os
import datetime
import matplotlib.pyplot as plt
import sys
import pandas as pd
import pickle
import copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

import sys
sys.path.append('../Libs')
sys.path.append('../KFold')

from util import *
from Preprocessing import *
from Model import *
from Evaluation import *

_, X, Y, _, _ = do_preprocess()

print(X.shape)
print(Y.shape)

# Load weights
weight_name = '10_Fold_Training_K_fold_EmoDB_Mel128_2022_Nov_16_18_49_31'
Learning_details_path = './Learning_Details/'
file_dir = os.listdir(Learning_details_path)

path = os.path.join(Learning_details_path, weight_name)
save_path = './Fold_Analyzer_result'
print(path)
print(save_path)


# Index details
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FullModel().to(device)
validation = ValidateKFoldModelOfDetails(path, model)
validation.ShowIndexDetails()

print('Selected Epoch number in each fold:')
print(validation.epoch_selected_in_fold_dict)

# Plot Training results
validation.PrintAllTrainingResultsInOnePicture(figsize=(20,7), save_path=save_path, moving_average_window_len=0)

# Plot Training results with smothing parameter: moving_average_window_len=4
validation.PrintAllTrainingResultsInOnePicture(figsize=(20,7), save_path=save_path, moving_average_window_len=4)

# Plot training result in a specific fold: Fold_4
validation.PlotTrainingResult('Fold_4', figsize=(20,7), save_path=save_path)

# Print Evaluation details in each fold
validation.EvalAndPrintModelInAllFold(X, Y)

# Confusion matrix
EMOTIONS = {'Neutral':0, 'Calm':1, 'Happy':2, 'Sad':3, 'Angry':4, 'Fearful':5, 'Disgust':6, 'Surprised':7}

# Drawing a mean confusion matrix
validation.PlotMeanConfusionMatrixInAllFold(X, Y, label=EMOTIONS, normalize='true', figsize=(17,7), save_path=save_path)

# Drawing a confusion matrix in a specific fold
validation.PlotFoldConfusionMatrix(X, Y, fold='Fold_0', label=EMOTIONS, normalize=None, figsize=(17,7), save_path=save_path)

# Print the training csv file
validation.CsvLoad()
