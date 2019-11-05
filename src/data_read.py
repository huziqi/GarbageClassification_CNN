# -*- coding: utf-8 -*-
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms

class Data_loader():
    def __init__(self, file_path):
        self.file_path= file_path

    def get_files(self):
        class_train= []
        label_train= []
        for first_category in os.listdir(self.file_path):
            for second_category in os.listdir(self.file_path+'/'+first_category):
                for third_category in os.listdir(self.file_path+'/'+first_category+'/'+second_category):
                    for pic in os.listdir(self.file_path+'/'+first_category+'/'+second_category+'/'+third_category+ '/Images_withoutrect'):
                        class_train.append(self.file_path+'/'+first_category+'/'+second_category+'/'+third_category
                                           + '/Images_withoutrect''/'+pic)
                        label_train.append(third_category)
        dict_label=set(label_train)
        self.label2indice= dict((c, i) for i, c in enumerate(dict_label))
        label_train= [self.label2indice[c] for c in label_train]
        temp= np.array([class_train, label_train])
        temp= temp.transpose()# 转置
        # shuffle the samples
        np.random.shuffle(temp)
        image_list= list(temp[:, 0])
        label_list= list(temp[:, 1])
        x_train, x_test, y_train, y_test= train_test_split(image_list, label_list, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def get_batches(self, image_path, label, resize, batch_size):
        image_batch=[]
        label_batch=[]
        for i in range(batch_size):
            index= np.random.randint(0,len(image_path))
            image=Image.open(image_path[index]).convert('RGB')
            transforms1= transforms.Compose([
                transforms.Scale(resize),
                transforms.ToTensor()
            ])
            image= transforms1(image)
            image_batch.append(image)
            label_batch.append(label[index])
        return image_batch, label_batch