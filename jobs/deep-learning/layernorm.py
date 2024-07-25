import torch
import torch.nn as nn

def layer_norm(x, dim, eps=1e-5):
    #layer_norm(dim=-1) x: [batch size, seq len, hidden size] => [batch size, seq len, 1]
    #batch_norm(dim=(0,1)) x: [batch size, seq len, hidden size] => [1, 1, hidden size]
    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)
    return (x - mean) / (std + eps)
    
class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        self.weight = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps
    
    def forward(self, x):
        mu = torch.mean(x, dim=-1, keepdim=True)
        sigma = torch.std(x, dim=-1, keepdim=True)
        return (x - mu) / (sigma + self.eps) * self.bias + self.weight

x = torch.rand((5, 10, 20))
y = layer_norm(x, dim=-1)
print(y.shape)
# print(y)