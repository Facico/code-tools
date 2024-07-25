import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

# gqa + kv cache
def group_query_attention(inputs, q_num_head, kv_num_head, past_kv=None):
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
    
    if past_kv is not None:
        keys = torch.cat([keys, past_kv[0]], dim=2)
        values = torch.cat([values, past_kv[1]], dim=2)
    past_kv = (keys, values)
    
    keys = repeat_kv(keys, repeat_num=kv_groups)
    values = repeat_kv(values, repeat_num=kv_groups)
    # [batch size, num head, seq len, attention head size]
    
    attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
    attention_scores /= math.sqrt(attention_head_size)
    attention_weights = F.softmax(attention_scores, dim=-1)
    outputs = torch.matmul(attention_weights, values)
    outputs = outputs.permute(0, 2, 1, 3).contiguous()
    outputs = outputs.view(batch_size, seq_len, hidden_size)
    
    return outputs, past_kv

batch_size, seq_len, hidden_size, num_head = 16, 20, 128, 5
X1 = torch.rand((batch_size, seq_len, hidden_size))
Y1, past_kv1 = group_query_attention(X1, 64, 8)
X2 = torch.rand((batch_size, seq_len, hidden_size))
Y2, past_kv2 = group_query_attention(X2, 64, 8, past_kv=past_kv1)

# 4hh+4h + 2h + 8hh+5h + 2h = 12hh + 13h
# 4hh+h+8hh+h=12hh+2h  +h
# 2hh+2hh/8 + 8hh*1.3 + 2h  =12.65hh+2h

#2bshv+l*(4*2bshh+2*2bssh + 2*2*4*bshh)=24bshh+4bssh

#2bsh+3*2bsh+2bssa+bssa+2bssa+2bsh+ bsh + 2bsh + 2*2*4bsh+bsh+2*2bsh=attn(11bsh+5bssa)+mlp(19bsh)+4=34bsh+5bssa