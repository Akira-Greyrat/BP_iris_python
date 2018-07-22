import pandas as pd
import numpy as np
from BP_version3 import BPNeuralNetwork
def dataReader():
   #读取数据
   data = pd.read_csv("D:\\workSpace\\TensorFlow\\iris\\iris.csv",header=0)
   #均值归一化nomalization
   for i in ['SepalLength','SepalWidth','PetalLength','PetalWidth']:
       data[i]=(data[i]-min(data[i]))/(max(data[i])-min(data[i]))
   return data
nn = BPNeuralNetwork([4,5,3],'sigmoid')
data=dataReader()
'''
当输入参数X为一个样本时增量学习
当输入参数X为多个样本时批量学习
'''
y=[]
for j in range(150):
    if data.loc[j][4]==1:
        y.append(np.array([1,0,0]))#列表转数组
               
    elif data.loc[j][4]==2:
        y.append(np.array([0,1,0]))
               
    else:
        y.append(np.array([0,0,1]))
               
nn.BPalgorithm(data.iloc[1:150,0:4],y,0.5,200)
print(nn.d)
