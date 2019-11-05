import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms, utils
import torch.utils.data as data
import torch
path='/home/huziqi/garbage_classification'
a=[]
b=[]
for i in range(50):
    a.append(i+2)
    b.append(i+2)
temp=np.array([a,b])
temp=temp.transpose()
c=list(temp[:,0])
d=list(temp[:,1])
x_train, x_test, y_train, y_test= train_test_split(c,d,test_size=0.3,random_state=0)
dataset= data.TensorDataset(x_train, y_train)
print(dataset)
# batch=DataLoader(x_train, batch_size=2, shuffle=True)
# print(batch[1])
# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)