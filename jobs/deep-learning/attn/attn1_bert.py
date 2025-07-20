import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

def multi_head_attention(inputs, num_head, mask=None, ep=-1e20):
    # X: [batch size, seq len, hidden size]
    def transpose_for_scores(x):
        new_shape = (x.shape[0], x.shape[1], num_head, x.shape[-1] // num_head)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    batch_size, seq_len, hidden_size = inputs.shape
    attention_head_size = hidden_size // num_head
    Wq = nn.Linear(hidden_size, attention_head_size * num_head)
    Wk = nn.Linear(hidden_size, attention_head_size * num_head)
    Wv = nn.Linear(hidden_size, attention_head_size * num_head)
    dropout = nn.Dropout(0.1)
    
    queries = Wq(inputs)
    keys = Wk(inputs)
    values = Wv(inputs)
    
    queries = transpose_for_scores(queries)
    keys = transpose_for_scores(keys)
    values = transpose_for_scores(values)
    # [batch size, num head, seq len, attention head size]
    
    attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
    attention_scores /= math.sqrt(attention_head_size)
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask==0, ep)
    attention_weights = F.softmax(attention_scores, dim=-1)
    attention_weights = dropout(attention_weights)
    
    outputs = torch.matmul(attention_weights, values)
    outputs = outputs.permute(0, 2, 1, 3).contiguous()
    outputs = outputs.view(batch_size, seq_len, hidden_size)
    return outputs

def selfoutput(inputs, num_head):
    batch_size, seq_len, hidden_size = inputs.shape
    Wo = nn.Linear(hidden_size, hidden_size)
    dropout = nn.Dropout(0.1)
    layernorm = nn.LayerNorm(hidden_size)
    hidden_states = multi_head_attention(inputs, num_head)
    output = inputs + dropout(layernorm(hidden_states))
    return output


batch_size, seq_len, hidden_size, num_head = 16, 20, 128, 5
X = torch.rand((batch_size, seq_len, hidden_size))
Y = selfoutput(X, 64)
print(Y)