import torch
import numpy as np
import random

class net():
    def __init__(self, dim):
        self.W = np.ones((dim,1))
        self.b = random.random()
    
    def forward(self, X, y):
        # X: [batch size, dim]
        pred = X@self.W + self.b
        loss = np.square(pred - y).sum() / X.shape[0]
        return pred, loss
    
    def backward(self, X, y, pred):
        batch_size, dim = len(X), len(X[0])
        dW = 2/batch_size*X.T@(pred - y)
        db = np.sum(2/batch_size*(pred - y))
        return dW, db
    
    def step(self, dW, db, lr):
        self.W -= lr * dW
        self.b -= lr * db
    
    def train(self, X, y, lr=1e-2, epoch=100):
        for _epoch in range(epoch):
            pred, loss =  self.forward(X, y)
            dW, db = self.backward(X, y, pred)
            self.step(dW, db, lr)
            print(f"epoch {_epoch} loss: {loss}")

        print(self.W, self.b)
Net = net(5)
X = np.matrix([[random.random() for j in range(5)] for i in range(4)])
y = np.matrix([[random.random()] for i in range(4)])

Net.train(X, y)