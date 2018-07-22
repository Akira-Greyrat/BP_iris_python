# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 21:07:39 2018

@author: Horizon
"""

import pandas as pd
import numpy as np
from BP_version3 import BPNeuralNetwork
def dataReader():
   #读取数据
   data = pd.read_csv("D:\\workSpace\\TensorFlow\\BP\\ANNeal.csv",header=0)
   return data
   
def dataReader_out():
   #读取数据
   data_out = pd.read_csv("D:\\workSpace\\TensorFlow\\BP\\ANNeal_output.csv",header=0)
   return data_out
   
nn = BPNeuralNetwork([38,20,5],'sigmoid')
data=dataReader()
data_out=dataReader_out()
'''
当输入参数X为一个样本时增量学习
当输入参数X为多个样本时批量学习
'''               
y=np.array(data_out)
nn.BPalgorithm(data,y,0.5,100)
print(nn.d)