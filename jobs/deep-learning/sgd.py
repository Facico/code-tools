import torch
import numpy as np
import random

class net():
    def __init__(self, dim):
        self.W = [random.random() for _ in range(dim)]
        self.b = random.random()
    
    def forward(self, X, y):
        # X: [batch size, dim]
        batch_size, dim = len(X), len(X[0])
        pred = []
        for i in range(batch_size):
            yy = self.b + sum([x*w for x, w in zip(X[i], self.W)])
            pred.append(yy)
        
        loss = 0
        for i in range(len(y)):
            loss += (y[i] - pred[i])**2/batch_size
        return pred, loss
    
    def backward(self, X, y, pred):
        batch_size, dim = len(X), len(X[0])
        dW = [0 for i in range(dim)]
        db = 0
        for i in range(batch_size):
            for j in range(dim):
                dW[j] += 2*(-y[i]+pred[i])*X[i][j]/batch_size
                db += 2*(-y[i]+pred[i])/batch_size
        return dW, db
    
    def step(self, dW, db, lr):
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
        self.b -= lr * db
    
    def train(self, X, y, lr=1e-3, epoch=100):
        for _epoch in range(epoch):
            pred, loss =  self.forward(X, y)
            dW, db = self.backward(X, y, pred)
            self.step(dW, db, lr)
            print(f"epoch {_epoch} loss: {loss}")

        print(self.W, self.b)
Net = net(5)
X = [[random.random() for j in range(5)] for i in range(4)]
y = [random.random() for i in range(4)]

Net.train(X, y)