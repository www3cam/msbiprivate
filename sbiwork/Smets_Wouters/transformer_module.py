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



#from torchsummary import summary

#import os
#os.system('export CUDA_VISIBLE_DEVICES=""')

numpos = 5

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, ntime, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        self.odd = False
        self.d_model = d_model
        self.max_len = max_len = ntime
        if d_model % 2 == 1:
            d_model = d_model + 1
            self.odd = True
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if self.odd:
            pe = pe[:,:-1]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        #x = x + self.pe[:x.size(0), :]
        bsz = x.shape[1]
        #x = x.transpose(0,1)
        x= x.to(torch.device("cpu"))
        return torch.cat([x,self.pe.repeat(1,bsz,1)], axis = 2)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, ntime, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(numpos, ntime, 0.)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp*ntime, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        #nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src))
                self.src_mask = mask
        else:
            self.src_mask = None
        src = src.reshape(-1,100,5)
        src = src.transpose(0,1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, None)
        output = output.transpose(0,1)
        output = output.flatten(start_dim=1)
        output = self.decoder(output)
        return output
    
class SummaryNetLSTM(nn.Module): 

    def __init__(self, timesteps = 100, covariates = 7, ydim = 4, classifier = False): 
        super().__init__()
        self.ts =timesteps
        self.cov = covariates
        self.classifier = classifier
        self.outfeat = 20
        self.ydim = 0
        if classifier:
            self.outfeat = 1
            self.ydim = ydim
        # 2D convolutional layer
        self.LSTM = nn.LSTM(self.cov, 2, 2, batch_first = True, bidirectional = True)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.flatten = nn.Flatten(start_dim=1)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc1 = nn.Linear(in_features=4*timesteps + ydim, out_features=64) 
        self.fc2 = nn.Linear(in_features=64, out_features=self.outfeat)
        self.relu = nn.ReLU()

    def forward(self,inputs : list):
        if self.classifier:
            x = inputs[1]
            y = inputs[0]
        else:
            x = inputs
        y = inputs[0]
        x= x.reshape(-1,self.ts,self.cov)
        h_0 = torch.zeros(4, x.size(0), 2, requires_grad=True) #hidden state
        c_0 = torch.zeros(4, x.size(0), 2, requires_grad=True) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.LSTM(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = self.flatten(output) #reshaping the data for Dense layer next
        if self.classifier:
            hn = torch.cat([hn,y],axis = 1)
        out = self.relu(hn)
        out = self.fc1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc2(out) #Final Output
        return out
    
    
class SummaryNetFeedForward(nn.Module): 

    def __init__(self, timesteps = 100, covariates = 7): 
        super().__init__()
        self.ts =timesteps
        self.cov = covariates
        # 2D convolutional layer
        self.LSTM = nn.LSTM(self.cov, 2, 2, batch_first = True, bidirectional = True)
        # Maxpool layer that reduces 32x32 image to 4x4
        self.flatten = nn.Flatten(start_dim=1)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc1 = nn.Linear(in_features=covariates*timesteps, out_features= 128) 
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=20)
        self.relu = nn.ReLU()

    def forward(self,x):
        x= x.reshape(-1,self.ts,self.cov)
        # Propagate input through LSTM
        hn = self.flatten(x) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc2(out) #Final Output
        out = self.relu(out) 
        out = self.fc3(out)

        return out

#embedding_net = SummaryNetLSTM()
#model = TransformerModel(20, 10, 100, 5, 64, 4, 0.0).to("cpu")
#print(embedding_net(torch.randn(1,100,5)))