import torch
import torch.nn as nn

def norm(x, dim, eps=1e-5):
    #layer_norm(dim=-1) x: [batch size, seq len, hidden size] => [batch size, seq len, 1]
    #batch_norm(dim=(0,1)) x: [batch size, seq len, hidden size] => [1, 1, hidden size]
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)
    return (x - mean) / (std + eps)

def rmsnorm(x):
    norm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+1e-6)
    (x / norm)
    
class RmsNorm(nn.Module):
    def __init__(self, dmodel, eps=1e-5):
        super(RmsNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dmodel))
        self.eps = eps
    def forward(self, x):
        norm = x / (torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dmodel, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.zeros(dmodel))
        self.bias = nn.Parameter(torch.ones(dmodel))
        self.eps = eps
    
    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True)
        return (x - mu) / (sigma + self.eps) * self.bias + self.w

class BatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
    
    def forward(self, x, training=True):
        if x.dim == 4: # 2d BN: N, C, H, W
            dim = (0, 2, 3)
            view_shape = (1, -1, 1, 1)
        else: # 1d BN: N, D
            dim = 0
            view_shape = (1, -1)
        if training:
            batch_mean = x.mean(dim=dim, keepdim=True)
            batch_var = x.var(dim=dim, keepdim=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.squeeze(0)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.squeeze(0)
            
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean.view(view_shape)
            var = self.running_var.view(view_shape)
        x_norm = (x-mean)/torch.sqrt(var + self.eps)
        out = x_norm * self.gemma + self.bias
        return out
        
x = torch.randn(8, 4)  # Batch=8, Features=4
bn = BatchNorm(num_features=4)
    
# net = RmsNorm(20)
# x = torch.rand((5, 10, 20))
# y = net(x)
# print(y.shape)
# print(y)