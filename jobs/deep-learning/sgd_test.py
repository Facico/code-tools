import random

class sgd():
    def __init__(self, dim):
        self.dim = dim
        self.W = [0 for i in range(dim)]
        self.b = 0
    def forward(self, x, y):
        batch_size = len(x)
        pred = []
        for i in range(batch_size):
            yy = self.b + sum([x*w for x, w in zip(x[i], self.W)])
            pred.append(yy)
        
        loss = 0
        for i in range(len(y)):
            loss += (y[i] - pred[i])**2/batch_size
        return pred, loss
    def backward(self, x, y, pred):
        batch_size, hidden_size = len(x), len(x[0])
        # y = wx+b loss=(y-pred)^2=(wx+b-pred)^2
        # dW = 2(wx+b-pred)x db=2(wx+b-pred)
        dW = [0 for i in range(hidden_size)]
        db = 0
        for i in range(batch_size):
            for j in range(hidden_size):
                dW[j]+=2*(pred[i]-y[i])*x[i][j]/batch_size
                db+=2*(pred[i]-y[i])/batch_size
        return dW, db
    def step(self, dW, db, lr):
        for i in range(self.dim):
            self.W[i] -= lr*dW[i]
        self.b -= lr*db
    def train(self, x, y, lr=1e-2, ep=100):
        for _ in range(ep):
            pred, loss = self.forward(x, y)
            dW, db = self.backward(x, y, pred)
            self.step(dW, db, lr)
            print(loss)

Net = sgd(5)
X = [[random.random() for j in range(5)] for i in range(4)]
y = [random.random() for i in range(4)]

Net.train(X, y)