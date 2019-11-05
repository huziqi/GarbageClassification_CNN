# -*- coding: utf-8 -*-

import data_read
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, num_chars, batch_size, seq_size):
        super(NN, self).__init__()
        self.num_chars= num_chars
        self.batch_size= batch_size
        self.seq_size= seq_size
        self.embedding= nn.Embedding(self.num_chars, 100)
        self.fc1 = nn.Sequential(
            nn.Linear(self.seq_size*self.num_chars*100, 250),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(250, self.num_chars)

    # 定义前向传播过程，输入为inputs
    def forward(self, inputs):
        inputs= torch.LongTensor(inputs)
        inputs= torch.reshape(inputs,(self.batch_size*self.seq_size, 1))
        inputs= torch.zeros(self.batch_size*self.seq_size, self.num_chars).scatter_(1, inputs, 1) # one-hot encoding
        inputs= torch.reshape(inputs, (self.batch_size, self.seq_size, self.num_chars))
        x= self.embedding(inputs.long())
        x= x.view(x.size(0), -1)
        x = self.fc1(x)
        outputs = self.fc2(x)
        return outputs

    def predict(self, inputs, temperature=1.):
        inputs = torch.LongTensor(inputs)
        inputs = torch.reshape(inputs, (self.seq_size, 1))
        inputs = torch.zeros(self.seq_size, self.num_chars).scatter_(1, inputs, 1)  # one-hot encoding
        inputs = torch.reshape(inputs, (1, self.seq_size, self.num_chars))
        x = self.embedding(inputs.long())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        predicts = self.fc2(x)
        prob = F.softmax(predicts/ temperature).detach().numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[0, :])])