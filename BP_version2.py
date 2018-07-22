import numpy as np
import pandas as pd

#激活函数及其导数
def tanh(x):
       return np.tanh(x)
def tanh_deri(x):
       return 1.0-np.tanh(x)*np.tanh(x)
def sigmoid(x):
       return 1/(1+np.exp(-x))
def sigmoid_deri(x):
       return sigmoid(x)*(1-sigmoid(x))

#三层前馈网络
class BPNeuralNetwork:
       #创建网络结构
       def __init__(self,layers,activation='sigmoid'):
              #layers表示各层神经元数目的list
              #确认激活函数
              if activation=='sigmoid':
                     self.activ=sigmoid
                     self.activ_deri=sigmoid_deri
              elif activation=='tanh':
                     self.activ=tanh
                     self.activ_deri=tanh_deri
              else:
                     print("input wrong")

              self.weights=[]
              self.d=0#正确个数

              #权值及阈值(w,b)随机初始化
              for i in range(len(layers)-1):
                     self.weights.append(np.random.random((layers[i]+1,layers[i+1])))
              print("随机初始化的权值矩阵:",self.weights)

       def BPalgorithm(self,X,y,step=0.3,times=50):
              #X，y为单样本,增量学习
              
              X=np.array(X)
              X=np.hstack((X,[1]))
              y=np.array(y)       
              for n in range(times):
                     
                     #正向传播
                     #隐层输入向量
                     z=np.dot(X,self.weights[0])
                     #隐层输出向量
                     a=self.activ(z)
                     a=np.hstack((a,[1]))
                     #预测向量
                     y_pred=self.activ(np.dot(a,self.weights[1]))
                     print(y)
                     print(y_pred)

                     for i in range(len(y)):
                            if y_pred[i]==max(y_pred) and y[i]==1:
                                   self.d=self.d+1
                     #Cost
                     error=sum((y-y_pred)**2)/2
                     
                     #反向传导
                     delt_out = np.array((y_pred-y)*self.activ_deri(y_pred))#输出层残差向量
                     #隐层残差
                     delt=[]
                     for p in range(len(a)-1):
                            delt.append(np.dot(delt_out,self.weights[1][p])*self.activ_deri(a[p]))
                            
                     #更新权值
                     for i in range(len(a)):
                            for j in range(len(y_pred)):
                                   self.weights[1][i][j]=self.weights[1][i][j]-step*delt_out[j]*a[i]
                     
                     for i in range(len(X)):
                            for j in range(len(a)-1):
                                    self.weights[0][i][j]= self.weights[0][i][j]-step*delt[j]*X[i]
              print(error)
              print(self.weights)
       
       def predict(self,x):
              x=np.array(x)
              a=x
              for i in range(len(self.weights)):
                     a=np.hstack((a,[1]))
                     a=self.activ(np.dot(a,self.weights[i]))
              return a
