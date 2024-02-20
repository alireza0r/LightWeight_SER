import torch
import torch.nn as nn
import numpy as np
from time import time
import os
import datetime
import matplotlib.pyplot as plt

class FeatureExtraction(nn.Module):
  def __init__(self):
    super().__init__()

    cnn = nn.Conv2d(1, 32, 3, 1, padding='same')
    bnormal = nn.BatchNorm2d(32)
    act = nn.ReLU()
    maxpool = nn.AvgPool2d(2,2)
    drop = nn.Dropout(0.4)
    self.Block1 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    cnn = nn.Conv2d(32, 64, 3, 1, padding='same')
    bnormal = nn.BatchNorm2d(64)
    act = nn.ReLU()
    maxpool = nn.AvgPool2d(2,2)
    drop = nn.Dropout(0.4)
    self.Block2 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    cnn = nn.Conv2d(64, 128, 3, 1, padding='same')
    bnormal = nn.BatchNorm2d(128)
    act = nn.ReLU()
    maxpool = nn.AvgPool2d(2,2)
    drop = nn.Dropout(0.4)
    self.Block3 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    cnn = nn.Conv2d(128, 256, 3, 1, padding='same')
    bnormal = nn.BatchNorm2d(256)
    act = nn.ReLU()
    maxpool = nn.AvgPool2d(2,2)
    drop = nn.Dropout(0.4)
    self.Block4 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    # cnn = nn.Conv2d(128, 256, 3, 1, padding='same')
    # bnormal = nn.BatchNorm2d(256)
    # act = nn.ReLU()
    # maxpool = nn.AvgPool2d(2,2)
    # drop = nn.Dropout(0.3)
    # self.Block5 = nn.Sequential(cnn, bnormal, act, maxpool, drop)

    # self.MHead4 = nn.MultiheadAttention(4*32, 4, dropout=0, batch_first=True)
    # self.MHead5 = nn.MultiheadAttention(2*16, 4, dropout=0, batch_first=True)

  def forward(self, x):
    #print(x.size()) # torch.Size([32, 1, 128, 563])

    x = self.Block1(x)

    #print(x.size()) # torch.Size([32, 16, 32, 140])

    x = self.Block2(x)

    #print(x.size()) # torch.Size([32, 32, 16, 70])

    x = self.Block3(x)

    #print(x.size()) # torch.Size([32, 64, 8, 35])

    x = self.Block4(x)
    #x_size = x.size()

    #x = self.Block5(x)
    # x_size = x.size()

    # xh = x.contiguous().view(x.size(0), x.size(1), -1)
    # xh, _ = self.MHead5(xh.clone(), xh.clone(), xh.clone())

    # xmax = torch.mean(xh, -1)
    # xmax = torch.unsqueeze(xmax, -1)

    # xh = torch.mul(xh, xmax)
    # x = x + xh.contiguous().view(x_size[0], x_size[1], x_size[2], x_size[3])

    # print(x_size)

    return x



class BottleNeck(nn.Module):
  def __init__(self, embed_dim=116, dropout=0.2, ch_features=256):
    super().__init__()

    self.MHead1 = nn.MultiheadAttention(embed_dim, 4, dropout=dropout, batch_first=True)
    self.BNorm1 = nn.BatchNorm1d(ch_features)
    self.MHead2 = nn.MultiheadAttention(embed_dim, 4, dropout=dropout, batch_first=True)
    self.BNorm2 = nn.BatchNorm1d(ch_features)
    self.MHead3 = nn.MultiheadAttention(embed_dim, 2, dropout=dropout, batch_first=True)
    self.BNorm3 = nn.BatchNorm1d(ch_features)
    self.MHead4 = nn.MultiheadAttention(embed_dim, 2, dropout=dropout, batch_first=True)
    self.BNorm4 = nn.BatchNorm1d(ch_features)

    self.max2 = nn.MaxPool1d(2,2)

  def forward(self, x):
    #print(x.size()) # torch.Size([32, 128, 68])

    xh, _ = self.MHead1(x, x, x)
    x = x + xh
    x = self.BNorm1(x)

    xh, _ = self.MHead2(x, x, x)
    x = x + xh
    x = self.BNorm2(x)
    #print(x.size()) # torch.Size([32, 128, 68])

    #x = self.max2(x)
    #print(x.size()) # torch.Size([32, 128, 34])

    # xh, _ = self.MHead3(x, x, x)
    # x = x + xh
    # x = self.BNorm3(x)

    # xh, _ = self.MHead4(x, x, x)
    # x = x + xh
    # x = self.BNorm4(x)
    #print(x.size()) # torch.Size([32, 128, 34])

    x = self.max2(x)
    x = self.max2(x)
    #print(x.size()) # torch.Size([32, 128, 17])
    return x


class Head(nn.Module):
  def __init__(self, n_input=256*29, n_out=8):
    super().__init__()

    act = nn.ReLU()

    F = nn.Linear(n_input, 128)
    BNorm = nn.BatchNorm1d(128)
    self.FC1 = nn.Sequential(F, BNorm, act)

    F = nn.Linear(128, 32)
    BNorm = nn.BatchNorm1d(32)
    self.FC2 = nn.Sequential(F, BNorm, act)

    F = nn.Linear(32, n_out)
    BNorm = nn.BatchNorm1d(n_out)
    self.FC3 = nn.Sequential(F, BNorm)

    self.Sig = nn.Sigmoid()

  def forward(self, x):
    #print(x.size()) # torch.Size([32, 2176])

    x = self.FC1(x)
    #print(x.size()) # torch.Size([32, 128])

    x = self.FC2(x)
    #print(x.size()) # torch.Size([32, 48])

    x = self.FC3(x)
    #print(x.size()) # torch.Size([32, 8])

    x = self.Sig(x)
    return x

class FullModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.featureextarction_model = FeatureExtraction()
    self.bottleneck_model = BottleNeck()
    self.head_model = Head()
    self.save_bottleneck = None

  def forward(self, x):
    # print(x.size())
    x = self.featureextarction_model(x)
    # print(x.size())
    x = torch.flatten(x, start_dim=-2)
    # print(x.size())
    x = self.bottleneck_model(x)
    self.save_bottleneck = x
    # print(x.size())
    x = torch.flatten(x, start_dim=-2)
    # print(x.size())
    x = self.head_model(x)
    # print(x.size())
    return x

