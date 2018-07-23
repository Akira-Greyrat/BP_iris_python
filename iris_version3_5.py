# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:10:57 2018

@author: Horizon
"""

import pandas as pd
import numpy as np
from BP_version3_5 import BPNeuralNetwork
def dataReader():
   #读取训练数据
   data = pd.read_csv("D:\\workSpace\\TensorFlow\\iris\\iris_training.csv",header=0)
   for i in ['SepalLength','SepalWidth','PetalLength','PetalWidth']:
       data[i]=(data[i]-min(data[i]))/(max(data[i])-min(data[i]))
   return data
   

def dataReader_test():
   #读取测试数据
   data_test = pd.read_csv("D:\\workSpace\\TensorFlow\\iris\\iris_test.csv",header=0)
   for i in ['SepalLength','SepalWidth','PetalLength','PetalWidth']:
       data_test[i]=(data_test[i]-min(data_test[i]))/(max(data_test[i])-min(data_test[i]))
   return data_test


nn = BPNeuralNetwork([4,5,3],'sigmoid')
data=dataReader()
data_test=dataReader_test()
'''
当输入参数X为一个样本时增量学习
当输入参数X为多个样本时批量学习
'''               
y=[]
for j in range(120):
    if data.loc[j][4]==0:
        y.append(np.array([1,0,0]))#列表转数组
               
    elif data.loc[j][4]==1:
        y.append(np.array([0,1,0]))
               
    else:
        y.append(np.array([0,0,1]))
               
y_test=[]
for j in range(30):
    if data_test.loc[j][4]==0:
        y_test.append(np.array([1,0,0]))#列表转数组
               
    elif data_test.loc[j][4]==1:
        y_test.append(np.array([0,1,0]))
               
    else:
        y_test.append(np.array([0,0,1]))
y=np.array(y)
y_test=np.array(y_test)
nn.BPalgorithm(data.iloc[0:120,0:4],y,0.3,200)
print(nn.d)

#测试集
nn.generalize(data_test.iloc[0:30,0:4],y_test)