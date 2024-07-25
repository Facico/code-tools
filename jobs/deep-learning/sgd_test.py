import random

class Net():
    def __init__(self, dim):
        self.W = [random.random() for i in range(dim)]
        self.b = random.random()
        self.dim = dim
        
    def forward(self, x, y):
        batch_size, dim = len(x), len(x[0])
        y_pred = []
        loss = 0
        for i in range(batch_size):
            y_pred.append(sum([self.W[j]*x[i][j] for j in range(dim)]) + self.b)
            loss += (y[i] - y_pred[i]) ** 2 / batch_size
        return y_pred, loss

    def backward(self, x, y, y_pred):
        batch_size, dim = len(x), len(x[0])
        dW = [0 for i in range(dim)]
        db = 0
        for i in range(batch_size):
            for j in range(dim):
                dW[j] += -2*(y[i]-y_pred[i])*x[i][j]/batch_size
            db += -2*(y[i]-y_pred[i])/batch_size
        return dW, db
    
    def step(self, dW, db, lr=1e-2):
        for i in range(self.dim):
            self.W[i] -= lr*dW[i]
        self.b -= lr*db
    
    def train(self, x, y, epoch=50):
        for _epoch in range(epoch):
            y_pred, loss = self.forward(x, y)
            dW, db = self.backward(x, y, y_pred)
            self.step(dW, db)
            print(loss)

Net = Net(5)
X = [[random.random() for j in range(5)] for i in range(4)]
y = [random.random() for i in range(4)]

Net.train(X, y)