from BP_version3 import BPNeuralNetwork
import numpy as np

nn = BPNeuralNetwork([2,5,1],'sigmoid')
x=[[-1,-1], [-1,1], [1,1], [1,-1]]
y = [[1],[0],[1],[0]]
nn.BPalgorithm(x,y,0.2,20000)
for i in [[-1,-1], [-1,1], [1,-1], [1,1]]:
       print(i, nn.predict(i))
