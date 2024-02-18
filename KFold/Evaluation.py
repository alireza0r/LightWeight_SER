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


#CSV structure:
#columns=['Epoch', 'Time', 'Acc', 'Loss', 'ValAcc', 'ValLoss', 'Weights saved', 'FoldName', 'FileName']

class ValidateKFoldModelOfDetails():
  def __init__(self, path, model):
    self.path = path
    self.csv_columns = ['Epoch', 'Time', 'Acc', 'Loss', 'ValAcc', 'ValLoss', 'Weights saved', 'FoldName', 'FileName']
    self.csv_file = self.CsvLoad()
    self.folds_name = self.GetFoldsName()

    self.best_weights, self.epoch_selected_in_fold_dict = self.FindBestWeightForEachFold()
    assert len(self.GetUnacceptableFold())==0, 'At least there is one unacceptable fold, fold without any saving weights'

    self.index_details_dict = self.LoadDataIndex()
    self.learning_result = self.LoadLearningResult()
    self.save_result_path = os.path.normpath(path).split('/')[-1]
    #print(self.save_result_path)

    self.model = model
    self.loss_function = nn.BCELoss(reduction='sum')

  def CsvLoad(self):
    csv_path = os.path.join(self.path, 'Reports.csv')
    with open(csv_path, 'r') as f:
      report_dataframe = pd.read_csv(f)
    return report_dataframe

  def LoadDataIndex(self):
    with open(os.path.join(path, 'index_dict_pickle'), 'rb') as f:
      data_index = pickle.load(f)
    return data_index

  def GetData(self, fold_name, data_name):
    data_number_dict = {'train':0, 'valid':1, 'test':2}

    #if len(self.index_details_dict['Folds'][self.folds_name[0]])!=3 and data_name=='valid':
    #  raise Exception('Your dataset is invalid...!, your dataset must have 2 or 3 len')

    if len(self.index_details_dict['Folds'][self.folds_name[0]]) == 2:
      train_index_in_fold = self.index_details_dict['Folds'][fold_name][0] # [0] -> train , [1] -> test
      valid_split_size_in_fold = self.index_details_dict['Valid_split_size']
      train_size = int(len(train_index_in_fold)*(1-valid_split_size_in_fold))

      if data_name=='test':
        return self.index_details_dict['Folds'][fold_name][1]
      elif data_name=='train':
        return train_index_in_fold[:train_size]
      elif data_name=='valid':
        return train_index_in_fold[train_size:]

    elif len(self.index_details_dict['Folds'][self.folds_name[0]]) == 3:
      return self.index_details_dict['Folds'][fold_name][data_number_dict[data_name]]
    else:
      raise Exception('Your dataset is invalid...!, your dataset must have 2 or 3 len')

  def LoadLearningResult(self):
    with open(os.path.join(path, 'result_dict_pickle'), 'rb') as f:
      result = pickle.load(f)
    return result

  def FindBestValidInFold(self, fold_name):
    new_csv_file = self.csv_file.copy()
    true_index = (new_csv_file['FoldName'] == fold_name) & (new_csv_file['Weights saved'] == True)

    # find best
    if new_csv_file[true_index].shape[0] == 0:
      return []
    if new_csv_file[true_index].shape[0] > 1:
      max_index = np.argmax(new_csv_file['ValAcc'][true_index])
      return new_csv_file[true_index].iloc[max_index]
    else:
      return new_csv_file[true_index].iloc[0]

  def GetFoldsName(self):
    return list(self.csv_file['FoldName'].value_counts().keys())

  def FindBestWeightForEachFold(self):
    best_weight_dict = {}
    epoch_selected_in_fold_dict = {}
    for f in self.folds_name:
      try:
        resut = self.FindBestValidInFold(f)
        best_weight_dict[f] = resut.FileName
        epoch_selected_in_fold_dict[f] = resut.Epoch
      except:
        best_weight_dict[f] = []
        epoch_selected_in_fold_dict[f] = -1
        print(f, ': there aren\'t any acceptable weights')

    return best_weight_dict, epoch_selected_in_fold_dict

  def GetUnacceptableFold(self):
    unacceptable_fold = []
    for f in self.best_weights.keys():
      if len(self.best_weights[f]) == 0:
        unacceptable_fold.append(f)
    return unacceptable_fold

  def LoadBestWeightIntoModelInSpecificFold(self, fold_name):
    #print('LoadBestWeightIntoModelInSpecificFold')
    weight_dir = self.best_weights[fold_name]

    self.model.load_state_dict(torch.load(weight_dir))
    return self.model

  def Eval(self, X, Y, batch_size=32):
    #print('Eval')
    self.model.eval()

    loss_list = []
    accuracy_list = []
    # forward
    index = np.arange(X.size(0))
    for b in range(int(np.ceil(X.size(0)/batch_size))):
      if b+1 != int(np.ceil(X.size(0)/batch_size)):
        batch_index = index[b*batch_size:(b+1)*batch_size]
      else:
        batch_index = index[b*batch_size:]
        if len(batch_index) == 1:
          break

      y_softmax = self.model(X[batch_index])
      loss = self.loss_function(y_softmax, Y[batch_index]) / y_softmax.size(0) # sum to mean
      loss_list.append(loss.item())

      accuracy = torch.sum(torch.argmax(Y[batch_index], dim=-1)==torch.argmax(y_softmax, dim=-1), dim=-1)/float(y_softmax.size(0))
      accuracy_list.append(accuracy.item())

    return sum(loss_list)/float(len(loss_list)), (sum(accuracy_list)/float(len(accuracy_list)))*100.0

  def Prediction(self, X):
    #print('Prediction')
    self.model.eval()

    y_pred_categorical = []
    for x in X:
      y_pred_categorical.append(torch.argmax(model(torch.unsqueeze(x, 0))).type(torch.int8))

    y_pred_categorical = torch.stack(y_pred_categorical, dim=0)
    return y_pred_categorical

  def PredAllDataInAllFold(self, X):
    #print('PredAllDataInAllFold')
    self.model.eval()

    result = {}
    # pred data with best weight in each fold.
    for fold in self.folds_name:
      # Load best weight in the specific fold
      self.LoadBestWeightIntoModelInSpecificFold(fold)

      #train_index_in_fold = self.index_details_dict['Folds'][fold][0] # [0] -> train , [1] -> test
      #valid_split_size_in_fold = self.index_details_dict['Valid_split_size']

      #train_size = int(len(train_index_in_fold)*(1-valid_split_size_in_fold))
      #train_index_in_fold, valid_index_in_fold = train_index_in_fold[:train_size], train_index_in_fold[train_size:]
      #test_index_in_fold = self.index_details_dict['Folds'][fold][1]

      train_index_in_fold = self.GetData(fold, 'train')
      valid_index_in_fold = self.GetData(fold, 'valid')
      test_index_in_fold = self.GetData(fold, 'test')

      train_categorical_pred = self.Prediction(X[train_index_in_fold])
      valid_categorical_pred = self.Prediction(X[valid_index_in_fold])
      test_categorical_pred = self.Prediction(X[test_index_in_fold])

      result[fold] = {'train_categorical_pred':train_categorical_pred,
                      'valid_categorical_pred':valid_categorical_pred,
                      'test_index_in_fold':test_categorical_pred}
    return result

  def EvalModelInAllFold(self, X, Y):
    #print('EvalModelInAllFold')
    result = {}
    # evaluate model with best weight in each fold.
    for fold in self.folds_name:
      #print(fold)
      # Load best weight in the specific fold
      self.LoadBestWeightIntoModelInSpecificFold(fold)

      #train_index_in_fold = self.index_details_dict['Folds'][fold][0] # [0] -> train , [1] -> test
      #valid_split_size_in_fold = self.index_details_dict['Valid_split_size']

      #train_size = int(len(train_index_in_fold)*(1-valid_split_size_in_fold))
      #train_index_in_fold, valid_index_in_fold = train_index_in_fold[:train_size], train_index_in_fold[train_size:]
      #test_index_in_fold = self.index_details_dict['Folds'][fold][1]

      train_index_in_fold = self.GetData(fold, 'train')
      valid_index_in_fold = self.GetData(fold, 'valid')
      test_index_in_fold = self.GetData(fold, 'test')

      loss_train, acc_train = self.Eval(X[train_index_in_fold], Y[train_index_in_fold], batch_size=16)
      loss_valid, acc_valid = self.Eval(X[valid_index_in_fold], Y[valid_index_in_fold], batch_size=16)
      loss_test, acc_test = self.Eval(X[test_index_in_fold], Y[test_index_in_fold], batch_size=16)

      result[fold] = {'loss_train':loss_train,
                      'acc_train':acc_train,
                      'loss_valid':loss_valid,
                      'acc_valid':acc_valid,
                      'loss_test':loss_test,
                      'acc_test':acc_test}

    return result

  def EvalAndPrintModelInAllFold(self, X, Y):
    evaluate_result = self.EvalModelInAllFold(X, Y)

    loss_train = []
    acc_train = []
    loss_valid = []
    acc_valid = []
    loss_test = []
    acc_test = []
    for fold in evaluate_result.keys():
      loss_train.append(evaluate_result[fold]['loss_train'])
      acc_train.append(evaluate_result[fold]['acc_train'])
      loss_valid.append(evaluate_result[fold]['loss_valid'])
      acc_valid.append(evaluate_result[fold]['acc_valid'])
      loss_test.append(evaluate_result[fold]['loss_test'])
      acc_test.append(evaluate_result[fold]['acc_test'])

      print('{:s}: Epoch:{:03d}, Loss Train: {:.03f}, Acc Train: {:.03f}, Loss Valid: {:.03f}, Acc Valid: {:.03f}, Loss Test: {:.03f}, Acc Test: {:.03f}'.format(fold,
                                                                                                                                                  self.epoch_selected_in_fold_dict[fold],
                                                                                                                                                  loss_train[-1],
                                                                                                                                                  acc_train[-1],
                                                                                                                                                  loss_valid[-1],
                                                                                                                                                  acc_valid[-1],
                                                                                                                                                  loss_test[-1],
                                                                                                                                                  acc_test[-1]))

    print('  Mean: Loss Train: {:.03f}, Acc Train: {:.03f}, Loss Valid: {:.03f}, Acc Valid: {:.03f}, Loss Test: {:.03f}, Acc Test: {:.03f}'.format(np.mean(loss_train),
                                                                                                                                                np.mean(acc_train),
                                                                                                                                                np.mean(loss_valid),
                                                                                                                                                np.mean(acc_valid),
                                                                                                                                                np.mean(loss_test),
                                                                                                                                                np.mean(acc_test)))

  def MeanConfusionMatrixInAllFold(self, X, Y, normalize='true'):
    pred_result = self.PredAllDataInAllFold(X=X)
    #print('EvalModelInAllFold')
    result = {}
    # evaluate model with best weight in each fold.
    for fold in self.folds_name:
      #train_size = len(self.index_details_dict['Folds'][fold][0])
      #valid_size = train_size * self.index_details_dict['Valid_split_size']
      #train_size -= valid_size

      #train_index_in_fold = self.index_details_dict['Folds'][fold][0]
      #train_index_in_fold, valid_index_in_fold = train_index_in_fold[:int(train_size)], train_index_in_fold[int(train_size):]
      #test_index_in_fold = self.index_details_dict['Folds'][fold][1]

      train_index_in_fold = self.GetData(fold, 'train')
      valid_index_in_fold = self.GetData(fold, 'valid')
      test_index_in_fold = self.GetData(fold, 'test')

      Y_tarin_in_fold = torch.argmax(Y[train_index_in_fold], -1).type(torch.int8)
      Y_valid_in_fold = torch.argmax(Y[valid_index_in_fold], -1).type(torch.int8)
      Y_test_in_fold = torch.argmax(Y[test_index_in_fold], -1).type(torch.int8)

      train_pred = pred_result[fold]['train_categorical_pred']
      valid_pred = pred_result[fold]['valid_categorical_pred']
      test_pred = pred_result[fold]['test_index_in_fold']

      cm_train = confusion_matrix(Y_tarin_in_fold.cpu().numpy(), train_pred.cpu().numpy(), normalize=normalize)
      cm_valid = confusion_matrix(Y_valid_in_fold.cpu().numpy(), valid_pred.cpu().numpy(), normalize=normalize)
      cm_test = confusion_matrix(Y_test_in_fold.cpu().numpy(), test_pred.cpu().numpy(), normalize=normalize)

      result[fold] = {'cm_train':cm_train, 'cm_valid':cm_valid, 'cm_test':cm_test}

    return result

  def PlotMeanConfusionMatrixInAllFold(self, X, Y, label, normalize='true', figsize=(22,7), save_path=''):
    cm_result_in_all_fold = self.MeanConfusionMatrixInAllFold(X=X, Y=Y, normalize=normalize)

    fig, ax = plt.subplots(1,3, figsize=figsize)
    plt.grid(False)

    cm = []
    for fold in self.folds_name:
      cm.append(cm_result_in_all_fold[fold]['cm_train'])

    cm = np.mean(cm, axis=0)
    print(cm.shape)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot(ax=ax[0], xticks_rotation=45, colorbar=False, values_format='.1f')
    ax[0].set_title('Mean, Train')

    cm = []
    for fold in self.folds_name:
      cm.append(cm_result_in_all_fold[fold]['cm_valid'])

    cm = np.mean(cm, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot(ax=ax[1], xticks_rotation=45, colorbar=False, values_format='.1f')
    ax[1].set_title('Mean, Valid')

    cm = []
    for fold in self.folds_name:
      cm.append(cm_result_in_all_fold[fold]['cm_test'])

    cm = np.mean(cm, axis=0)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot(ax=ax[2], xticks_rotation=45, colorbar=False, values_format='.1f')
    ax[2].set_title('Mean, Test')

    if save_path != '':
      save_path = os.path.join(save_path, self.save_result_path)
      os.makedirs(save_path, exist_ok=True)
      plt.savefig(os.path.join(save_path, 'Mean_Confusion_matrix.svg'), format="svg")

  def PlotFoldConfusionMatrix(self, X, Y, fold, label, normalize='true', figsize=(22,7), save_path=''):
    pred_result = self.PredAllDataInAllFold(X=X)

    #train_size = len(self.index_details_dict['Folds'][fold][0])
    #valid_size = train_size * self.index_details_dict['Valid_split_size']
    #train_size -= valid_size

    #train_index_in_fold = self.index_details_dict['Folds'][fold][0]
    #train_index_in_fold, valid_index_in_fold = train_index_in_fold[:int(train_size)], train_index_in_fold[int(train_size):]
    #test_index_in_fold = self.index_details_dict['Folds'][fold][1]

    train_index_in_fold = self.GetData(fold, 'train')
    valid_index_in_fold = self.GetData(fold, 'valid')
    test_index_in_fold = self.GetData(fold, 'test')

    Y_tarin_in_fold = torch.argmax(Y[train_index_in_fold], -1).type(torch.int8)
    Y_valid_in_fold = torch.argmax(Y[valid_index_in_fold], -1).type(torch.int8)
    Y_test_in_fold = torch.argmax(Y[test_index_in_fold], -1).type(torch.int8)

    train_pred = pred_result[fold]['train_categorical_pred']
    valid_pred = pred_result[fold]['valid_categorical_pred']
    test_pred = pred_result[fold]['test_index_in_fold']

    print('Train ACC: {:.03f}%'.format(torch.sum(train_pred == Y_tarin_in_fold)/train_pred.size(0)*100))
    print('Valid ACC: {:.03f}%'.format(torch.sum(valid_pred == Y_valid_in_fold)/valid_pred.size(0)*100))
    print(' Test ACC: {:.03f}%'.format(torch.sum(test_pred == Y_test_in_fold)/test_pred.size(0)*100))

    fig, ax = plt.subplots(1,3, figsize=figsize)
    plt.grid(False)

    cm = confusion_matrix(Y_tarin_in_fold.cpu().numpy(), train_pred.cpu().numpy(), normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot(ax=ax[0], xticks_rotation=45, colorbar=False)
    ax[0].set_title('Train ' + fold)

    cm = confusion_matrix(Y_valid_in_fold.cpu().numpy(), valid_pred.cpu().numpy(), normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot(ax=ax[1], xticks_rotation=45, colorbar=False)
    ax[1].set_title('Valid ' + fold)

    cm = confusion_matrix(Y_test_in_fold.cpu().numpy(), test_pred.cpu().numpy(), normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    disp.plot(ax=ax[2], xticks_rotation=45, colorbar=False)
    ax[2].set_title('Test ' + fold)

    if save_path != '':
      save_path = os.path.join(save_path, self.save_result_path)
      os.makedirs(save_path, exist_ok=True)
      plt.savefig(os.path.join(save_path, fold + '.svg'), format="svg")

  def ShowIndexDetails(self):
    print('Train + Valid data size: {:d}'.format(len(self.index_details_dict['Folds']['Fold_0'][0])))
    print('Test data size: {:d}'.format(len(self.index_details_dict['Folds']['Fold_0'][1])))
    print('Valid split size: {:f}'.format(self.index_details_dict['Valid_split_size']))

  def PlotTrainingResult(self, fold, figsize=(20,7), save_path=''):
    index_in_fold = self.csv_file['FoldName'] == fold

    epoch_max_val_in_fold = np.argmax(self.csv_file['ValAcc'].loc[index_in_fold])
    epoch_max_val_in_index = self.csv_file['Epoch'].loc[index_in_fold].index[epoch_max_val_in_fold]

    print('{:s}, Selected Epoch: {:d}'.format(fold, epoch_max_val_in_fold))
    print('{:s}, Train  Acc.: {:.03f}%'.format(fold, float(self.csv_file['Acc'].iloc[epoch_max_val_in_index])))
    print('{:s}, Train Loss.: {:.03f}'.format(fold, float(self.csv_file['Loss'].iloc[epoch_max_val_in_index])))
    print('{:s}, Valid  Acc.: {:.03f}%'.format(fold, float(self.csv_file['ValAcc'].iloc[epoch_max_val_in_index])))
    print('{:s}, Valid Loss.: {:.03f}'.format(fold, float(self.csv_file['ValLoss'].iloc[epoch_max_val_in_index])))

    train_acc = self.csv_file['Acc'].loc[index_in_fold]
    train_loss = self.csv_file['Loss'].loc[index_in_fold]
    val_acc = self.csv_file['ValAcc'].loc[index_in_fold]
    val_loss = self.csv_file['ValLoss'].loc[index_in_fold]

    # Define interpolators.
    x = np.linspace(0, val_acc.shape[0]-1, num=val_acc.shape[0]*2)
    spl = make_interp_spline(np.arange(val_acc.shape[0]), val_acc, k=3)

    fig, ax = plt.subplots(1,2, figsize=figsize)

    ax[0].plot(np.arange(train_acc.shape[0]), train_acc, label='Train')
    ax[0].plot(np.arange(val_acc.shape[0]), val_acc, label='Valid')
    #ax[0].plot(x, spl(x), label='Smooth Valid')
    ax[0].scatter(epoch_max_val_in_fold, self.csv_file['ValAcc'].iloc[epoch_max_val_in_index], color='r', label='Select Point', zorder=21)
    ax[0].axhline(y = self.csv_file['ValAcc'].iloc[epoch_max_val_in_index], color = '0.8', linestyle = '--')
    ax[0].axvline(x = epoch_max_val_in_fold, color = '0.8', linestyle = '--')
    ax[0].set_title('Accuray , {:s} , Epoch selected={:d} , Train Acc.={:.03f}% , Validation Acc.={:.03f}%'.format(str(fold),
                                                                                                                  self.epoch_selected_in_fold_dict[fold],
                                                                                                                  float(self.csv_file['Acc'].iloc[epoch_max_val_in_index]),
                                                                                                                  float(self.csv_file['ValAcc'].iloc[epoch_max_val_in_index])))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuray %')
    ax[0].legend()

    ax[1].plot(np.arange(train_loss.shape[0]), train_loss, label='Train')
    ax[1].plot(np.arange(val_loss.shape[0]), val_loss, label='Valid')
    ax[1].scatter(epoch_max_val_in_fold, self.csv_file['ValLoss'].iloc[epoch_max_val_in_index], color='r', label='Select Point', zorder=21)
    ax[1].axhline(y = self.csv_file['ValLoss'].iloc[epoch_max_val_in_index], color = '0.8', linestyle = '--')
    ax[1].axvline(x = epoch_max_val_in_fold, color = '0.8', linestyle = '--')
    ax[1].set_title('Loss , {:s} , Epoch selected={:d} , Train Loss={:.03f} , Validation Loss={:.03f}'.format(str(fold),
                                                                                                               self.epoch_selected_in_fold_dict[fold],
                                                                                                               float(self.csv_file['Loss'].iloc[epoch_max_val_in_index]),
                                                                                                               float(self.csv_file['ValLoss'].iloc[epoch_max_val_in_index]),
                                                                                                               ))
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    if save_path != '':
      save_path = os.path.join(save_path, self.save_result_path)
      os.makedirs(save_path, exist_ok=True)
      plt.savefig(os.path.join(save_path, fold + '_Acc_Loss_TrainingResult.svg'), format="svg")

    plt.show()

  def PrintAllTrainingResultsInOnePicture(self, figsize=(20,7), save_path='', moving_average_window_len=0):
    fig, ax = plt.subplots(1,2, figsize=figsize)
    for fold in self.folds_name:
      index_in_fold = self.csv_file['FoldName'] == fold

      train_loss = self.MovingAverage(self.csv_file['Loss'].loc[index_in_fold].values, moving_average_window_len)
      val_acc = self.MovingAverage(self.csv_file['ValAcc'].loc[index_in_fold].values, moving_average_window_len)
      epoch_max_val_in_index = np.argmax(val_acc)

      # Validation Accuracy
      #ax[0].plot(np.arange(train_acc.shape[0]), train_acc, label='Train')
      ax[0].plot(np.arange(len(val_acc)), val_acc, label=fold)
      #ax[0].plot(x, spl(x), label='Smooth Valid')
      ax[0].scatter(epoch_max_val_in_index, val_acc[epoch_max_val_in_index], color='r', zorder=21)
      ax[0].axhline(y = val_acc[epoch_max_val_in_index], color = '0.8', linestyle = '--', zorder=0)
      ax[0].axvline(x = epoch_max_val_in_index, color = '0.8', linestyle = '--', zorder=0)

      # Train loss
      ax[1].plot(np.arange(len(train_loss)), train_loss, label=fold)
      #ax[1].plot(np.arange(val_loss.shape[0]), val_loss, label='Valid')
      ax[1].scatter(epoch_max_val_in_index, train_loss[epoch_max_val_in_index], color='r', zorder=21)
      ax[1].axhline(y = train_loss[epoch_max_val_in_index], color = '0.8', linestyle = '--', zorder=0)
      ax[1].axvline(x = epoch_max_val_in_index, color = '0.8', linestyle = '--', zorder=0)

    ax[0].set_title('Accuray Validation dataset , Moving average length: ' + str(moving_average_window_len))
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuray %')
    ax[0].legend(loc='lower left')

    ax[1].set_title('Loss train dataset , Moving average length: ' + str(moving_average_window_len))
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend(loc='upper left')

    if save_path != '':
      save_path = os.path.join(save_path, self.save_result_path)
      os.makedirs(save_path, exist_ok=True)
      plt.savefig(os.path.join(save_path, 'VaLAcc_Loss_AllFold_TrainingResult' + 'Moving_avg_' + str(moving_average_window_len) + '.svg'), format="svg")

    plt.show()

  def MovingAverage(self, x, window_len):
    assert window_len >= 0, 'Moving average length must be greater than zero'
    if window_len == 0:
      return x

    x_avg = []
    for s_point in range(len(x)-window_len+1):
      x_avg.append(np.mean(x[s_point:s_point + window_len]))
    return x_avg
