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

W=np.random.random((4,5))#输入层权值二维数组随机初始化
V=np.random.random((5,3))#隐层权值
global a1,a2,a3,a4,a5

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
                     global a1,a2,a3,a4,a5
                     #隐层的输入
                     z11=sum(x*W[:,0])
                     z12=sum(x*W[:,1])
                     z13=sum(x*W[:,2])
                     z14=sum(x*W[:,3])
                     z15=sum(x*W[:,4])
                     #隐层输出
                     a1=sigmoid(z11)
                     a2=sigmoid(z12)
                     a3=sigmoid(z13)
                     a4=sigmoid(z14)
                     a5=sigmoid(z15)
                     #输出层输出
                     y1=sigmoid(a1*V[0,0]+a2*V[1,0]+a3*V[2,0]+a4*V[3,0]+a5*V[4,0])
                     y2=sigmoid(a1*V[0,1]+a2*V[1,1]+a3*V[2,1]+a4*V[3,1]+a5*V[4,1])
                     y3=sigmoid(a1*V[0,2]+a2*V[1,2]+a3*V[2,2]+a4*V[3,2]+a5*V[4,2])
                     y_pred=np.array([y1,y2,y3])
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
              a=np.array([a1,a2,a3,a4,a5])
              delt1 = sum(delt_out*V[0])*a1*(1-a1)
              delt2 = sum(delt_out*V[1])*a2*(1-a2)
              delt3 = sum(delt_out*V[2])*a3*(1-a3)
              delt4 = sum(delt_out*V[3])*a4*(1-a4)
              delt5 = sum(delt_out*V[4])*a5*(1-a5)
              delt = np.array([delt1,delt2,delt3,delt4,delt5])

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



