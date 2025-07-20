import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
#gqa
def group_query_attention(inputs, q_num_head, kv_num_head, mask=None, ep=-1e20):
    # X: [batch size, seq len, hidden size]
    def transpose_for_scores(x, num_head):
        new_shape = (x.shape[0], x.shape[1], num_head, x.shape[-1] // num_head)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    
    def repeat_kv(x, repeat_num):
        # x: [batch size, kv num head, seq len, head size] => [batch size, kv num head * repeat_num, seq len, head size]
        batch_size, kv_num_head, seq_len, head_size = x.shape
        if repeat_num == 1:
            return x
        x = x[:, :, None, :, :].expand(batch_size, kv_num_head, repeat_num, seq_len, head_size)
        return x.reshape(batch_size, kv_num_head * repeat_num, seq_len, head_size)
    batch_size, seq_len, hidden_size = inputs.shape
    attention_head_size = hidden_size // q_num_head
    kv_groups = q_num_head // kv_num_head
    Wq = nn.Linear(hidden_size, attention_head_size * q_num_head) #64
    Wk = nn.Linear(hidden_size, attention_head_size * kv_num_head) #8
    Wv = nn.Linear(hidden_size, attention_head_size * kv_num_head)
    
    queries = Wq(inputs)
    keys = Wk(inputs)
    values = Wv(inputs)
    
    queries = transpose_for_scores(queries, q_num_head)
    keys = transpose_for_scores(keys, kv_num_head)
    values = transpose_for_scores(values, kv_num_head)
    
    keys = repeat_kv(keys, repeat_num=kv_groups)
    values = repeat_kv(values, repeat_num=kv_groups)
    # [batch size, num head, seq len, attention head size]
    
    attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
    attention_scores /= math.sqrt(attention_head_size)
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask==0, ep)
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    outputs = torch.matmul(attention_weights, values)
    outputs = outputs.permute(0, 2, 1, 3).contiguous()
    outputs = outputs.view(batch_size, seq_len, hidden_size)
    return outputs

batch_size, seq_len, hidden_size, num_head = 16, 20, 128, 5
X = torch.rand((batch_size, seq_len, hidden_size))
Y = group_query_attention(X, 64, 8)
print(Y)