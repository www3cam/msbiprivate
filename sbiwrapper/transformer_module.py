#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:08:00 2021

@author: cameron
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
class SummaryNetFeedForward(nn.Module): 

    def __init__(self, timesteps, covariates, hiddendim = 128, embeddingdim = 20): 
        super().__init__()
        self.ts =timesteps
        self.cov = covariates
        self.flatten = nn.Flatten(start_dim=1)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc1 = nn.Linear(in_features=covariates*timesteps, out_features= hiddendim) 
        self.fc2 = nn.Linear(in_features=hiddendim, out_features=hiddendim)
        self.fc3 = nn.Linear(in_features=hiddendim, out_features=embeddingdim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x= x.reshape(-1,self.ts,self.cov)
        # Propagate input through LSTM
        hn = self.flatten(x) #reshaping the data for Dense layer next
        out = self.fc1(hn) #first Dense
        out = self.relu(out) #relu
        out = self.fc2(out) #Final Output
        out = self.relu(out) 
        out = self.fc3(out)

        return out
