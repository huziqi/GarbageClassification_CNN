# -*- coding: utf-8 -*-

import data_read
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, num_classes):
        self.num_classes= num_classes
        super(NN, self).__init__()
        self.norm  = nn.LocalResponseNorm(size=5)
        self.drop  = nn.Dropout()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.full6 = nn.Linear(79872, 4096)
        self.full7 = nn.Linear(4096, 4096)
        self.full8 = nn.Linear(4096, self.num_classes)


    # 定义前向传播过程，输入为inputs
    def forward(self, inputs):
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