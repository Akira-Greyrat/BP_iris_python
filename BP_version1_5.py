# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 08:33:02 2018

@author: Horizon
"""
import pandas as pd
import numpy as np
from math import e
#数据读取和处理
data = pd.read_csv("D:\\workSpace\\TensorFlow\\iris\\iris.csv",header=0)
#数据集的基本操作
print(data.head())
print(data.columns)
print(data['Species'])#输出指定索引的一列
print(data.loc[0][0:4])#数组第0行前4个数
global d,q,s
d,q,s=4,5,3

W=np.random.random((d,q))#输入层权值二维数组随机初始化
V=np.random.random((q,s))#隐层权值

#激活函数
def sigmoid(x):
       return 1/(1+e**(-x))

#数组的基本操作
print(W[:,1])#第二列
print(W[0])#第一行
print(V[:,0])
print(W[0]*V[:,0])#对应位相乘

#均值归一化nomalization
data['SepalLength']=(data['SepalLength']-min(data['SepalLength']))/(max(data['SepalLength'])-min(data['SepalLength']))
data['SepalWidth']=(data['SepalWidth']-min(data['SepalWidth']))/(max(data['SepalWidth'])-min(data['SepalWidth']))
data['PetalLength']=(data['PetalLength']-min(data['PetalLength']))/(max(data['PetalLength'])-min(data['PetalLength']))
data['PetalWidth']=(data['PetalWidth']-min(data['PetalWidth']))/(max(data['PetalWidth'])-min(data['PetalWidth']))

for m in range(50):
       #遍历全部样本
       for p in range(150):
              global a,z,delt
              a=np.zeros(q)
              z=np.zeros(q)
              delt=np.zeros(q)
              #读取样本
              x=np.array(data.loc[p][0:4])
              if data.loc[p][4]==1:
                     y=np.array([1,0,0])#列表转数组
                     print(y)
              elif data.loc[p][4]==2:
                     y=np.array([0,1,0])
                     print(y)
              else:
                     y=np.array([0,0,1])
                     print(y)

              #前向传播
              def Forward(x):
                     #隐层的输入
                     #隐层输出
                     y_pred=np.zeros(3)
                     for m in range(5):
                         z[m]=sum(x*W[:,m])
                         a[m]=sigmoid(z[m])
                     #输出层输出
                     for n in range(3):
                         y_pred[n]=sigmoid(sum(a*V[:,n]))
                     print(y_pred)
                     return y_pred

              y_pred = Forward(x)
              #单样本损失函数
              def Error():
                     E=sum((y-y_pred)**2)/2
                     return E

              #后向传导
              E = Error();
              print("当前误差:",E)
              delt_out = (y_pred-y)*y_pred*(1-y_pred)#输出层残差向量

              #隐层残差
              for f in range(5):
                  delt[f]=sum(delt_out*V[f])*a[f]*(1-a[f])

              #学习率
              step=0.5
              #更新权值
              for i in range(5):
                     for j in range(3):
                            V[i,j]=V[i,j]-step*delt_out[j]*a[i]
              #print(V)

              for i in range(4):
                     for j in range(5):
                            W[i,j]=W[i,j]-step*delt[j]*x[i]
              #print(W)




