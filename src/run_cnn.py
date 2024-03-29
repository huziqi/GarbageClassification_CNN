#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
import data_read
import Net_structure
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path= "/home/huziqi/Pictures/dataset"
batch_size= 20
learning_rate= 0.001
EPOCH= 450
resize=(227,227)
########load train dataset
trainloader, num_classes= data_read.loadtraindata(path,batch_size, resize)
print("total classes: ", num_classes)
data_fromfolder=data_read.loaddatafromfolder()
dataset=data_fromfolder
########load test dataset
testdataloader= data_read.loadtestdata(path,batch_size, resize)
testdatafromfolder = data_read.loadtestdatafromfolder()
testdataset=testdatafromfolder

net = Net_structure.Alexnet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
optimizer= optim.Adam(net.parameters(), lr= learning_rate)
optimizer2= optim.SGD(net.parameters(),lr=learning_rate, momentum=0.9)


# 训练
if __name__ == "__main__":
    start_time= time.time()
    torch.cuda.empty_cache()
    for epoch in range(EPOCH):
        print('epoch{}'.format(epoch+1))
        sum_loss = 0.0
        sum_acc= 0.0
        for inputs, labels in tqdm(dataset):
            inputs = Variable(inputs)
            labels = Variable(labels)
            inputs=inputs.to(device)
            labels=labels.to(device)
            # 梯度清零
            optimizer2.zero_grad()
            # forward + backward
            outputs = net(inputs)
            labels=labels.long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer2.step()
            end_time= time.time()
            sum_loss+=float(loss.data.item())
            pred=torch.max(outputs,1)[1]
            train_correct= (pred==labels).sum()
            sum_acc+=float(train_correct.data.item())
        end_time=time.time()
        print('Train Loss: {:.6f}, Acc: {:.6f}, cost time: {:.2f}'.format(sum_loss/(len(dataset.dataset)), sum_acc/(len(dataset.dataset)),end_time-start_time))
    end_time=time.time()
    print('total cost time is: %.2f'%(start_time-end_time))

    torch.save(net.state_dict(),'/home/huziqi/garbage_classification/model/13145pic_28cla_20bs_VGG16_%d.pt'%EPOCH)
    #fout = open(output_path + str(EPOCH) + "_word_based_output.txt", "w")

    start_time=time.time()
    test_acc=0
    for images, labels in tqdm(testdatafromfolder):
        images, labels= Variable(images).to(device), Variable(labels).to(device)
        outputs = net(images)
        labels = labels.long()
        pred = torch.max(outputs, 1)[1]
        train_correct = (pred == labels).sum()
        test_acc += float(train_correct.data.item())
    end_time = time.time()
    print('Test data Acc: {:.6f}, cost time: {:.2f}'.format(test_acc / (len(testdataset.dataset)), end_time-start_time))