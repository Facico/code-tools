import torch
import torch.nn as nn
import torch.nn.functional as F

def swiglu(x):
    return x*F.sigmoid(x)
def MLP(x):
    batch_size, seq_len, hidden_size = x.shape
    W1 = nn.Linear(hidden_size, 8*hidden_size//3)
    W2 = nn.Linear(hidden_size, 8*hidden_size//3)
    W3 = nn.Linear(8*hidden_size//3, hidden_size)
    return W3(swiglu(W1(x))*W2(x))

x = torch.randn((16,10,128))
y = MLP(x)
print(y)