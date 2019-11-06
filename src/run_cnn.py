#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
import data_read
import Net_structure
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path= "/home/huziqi/baidunetdiskdownload"
dataloader = data_read.Data_loader(path)
batch_size= 4
learning_rate= 0.001
EPOCH= 5

net = Net_structure.NN(dataloader.get_num_classes()).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer= optim.Adam(net.parameters(), lr= learning_rate)
x_train, x_test, y_train, y_test= dataloader.get_files()

# 训练
if __name__ == "__main__":
    start_time= time.time()
    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        inputs, labels = dataloader.get_batches(x_train, y_train,227,4)

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, torch.LongTensor(labels))
        loss.backward()
        optimizer.step()
        end_time= time.time()
        print('the %d epoch loss: %.03f, total running time: %.2f s'% (epoch, loss.item(),end_time-start_time))


    #torch.save(net.state_dict(),'/home/guohf/AI_tutorial/ch8/model/oldman_wordbased_%d.pt'%EPOCH)
    fout = open(output_path + str(EPOCH) + "_word_based_output.txt", "w")