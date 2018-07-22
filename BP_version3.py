import numpy as np

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
              self.y_pred=0
              self.error=0

              #权值及阈值(w,b)随机初始化
              for i in range(len(layers)-1):
                     self.weights.append(np.random.random((layers[i]+1,layers[i+1])))
              self.weights_after=self.weights
              print("随机初始化的权值矩阵:",self.weights)
              
       def Forward(self,X,y):#单样本前向传播
              X=np.array(X)
              y=np.array(y)
              #正向传播
              #隐层输入向量
              z=np.dot(X,self.weights[0])
              #隐层输出向量
              a=self.activ(z)
              a=np.hstack((a,[1]))
              #预测向量
              self.y_pred=self.activ(np.dot(a,self.weights[1]))
              #print(y)
              #print(self.y_pred)

              for i in range(len(y)):
                     if self.y_pred[i]==max(self.y_pred) and y[i]==1:
                            self.d=self.d+1
              #Cost
              self.error=self.error+sum((y-self.y_pred)**2)/2
              return a

       def DeltUpdata(self,X,y,a,step):
              #反向传导
              delt_out = np.array((self.y_pred-y)*self.activ_deri(self.y_pred))#输出层残差向量
              #隐层残差
              delt=[]
              for p in range(len(a)-1):
                     delt.append(np.dot(delt_out,self.weights_after[1][p])*self.activ_deri(a[p]))
                     
              #更新权值
              for i in range(len(a)):
                     for j in range(len(self.y_pred)):
                            self.weights_after[1][i][j]=self.weights_after[1][i][j]-step*delt_out[j]*a[i]
              
              for i in range(len(X)):
                     for j in range(len(a)-1):
                            self.weights_after[0][i][j]= self.weights_after[0][i][j]-step*delt[j]*X[i]

       def weightsUpdata(self):
              self.weights=self.weights_after
              
       def BPalgorithm(self,X,y,step=0.3,times=50):
              #X，y为样本,批量学习
              for n in range(times):
                     self.error=0
                     for i in range(len(X)):
                            X=np.array(X)
                            X_s=X[i]
                            X_s=np.hstack((X_s,[1]))
                            y_s=y[i]
                            a=self.Forward(X_s,y_s)
                            self.DeltUpdata(X_s,y_s,a,step)
                     
                     self.weightsUpdata()
                     print("目前误差:",self.error)
       
       def predict(self,x):
              x=np.array(x)
              a=x
              for i in range(len(self.weights)):
                     a=np.hstack((a,[1]))
                     a=self.activ(np.dot(a,self.weights[i]))
              return a
