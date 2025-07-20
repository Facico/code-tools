import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_sin_pe(x):
    batch_size, head_num, seq_len, d = x.shape
    d_id = torch.pow(10000, -2*torch.arange(0, d//2) / d)
    s_id = torch.arange(0, seq_len).unsqueeze(-1)
    pe = s_id*d_id
    pe = torch.stack([torch.sin(pe), torch.cos(pe)], -1)
    pe = pe.repeat(batch_size, head_num, 1, 1, 1)
    return pe.reshape(batch_size, head_num, seq_len, d)
def rope(x):
    pe = get_sin_pe(x)
    cos_pe = pe[...,1::2].repeat_interleave(2, dim=-1)
    sin_pe = pe[...,::2].repeat_interleave(2, dim=-1)
    x2 = torch.stack([-x[...,1::2], x[...,::2]], dim=-1).reshape(x.shape)
    return x*cos_pe+x2*sin_pe

def multi_head_attention(x, q_num_head, kv_num_head, mask=None, kv_cache=None):
    def repeat_kv(x, repeat_num):
        if repeat_num == 1:
            return x
        batch_size, head_num, seq_len, head_size = x.shape
        x = x[:, :, None, :, :].expand(batch_size, head_num, repeat_num, seq_len, head_size)
        return x.reshape(batch_size, head_num*repeat_num, seq_len, head_size)
    batch_size, seq_len, hidden_size = x.shape
    head_size = hidden_size // q_num_head
    Wq = nn.Linear(hidden_size, q_num_head*head_size)
    Wk = nn.Linear(hidden_size, kv_num_head*head_size)
    Wv = nn.Linear(hidden_size, kv_num_head*head_size)
    Wo = nn.Linear(hidden_size, hidden_size)
    dropout = nn.Dropout(0.1)
    
    q = Wq(x).view(batch_size, seq_len, q_num_head, head_size).transpose(1,2)
    k = Wk(x).view(batch_size, seq_len, kv_num_head, head_size).transpose(1,2)
    v = Wv(x).view(batch_size, seq_len, kv_num_head, head_size).transpose(1,2)
    
    q, k = rope(q), rope(k)

    if kv_cache is not None:
        k = torch.cat([kv_cache[0], k], dim=2)
        v = torch.cat([kv_cache[1], v], dim=2)
    kv_cache = (k, v)
    
    k, v = repeat_kv(k, q_num_head//kv_num_head), repeat_kv(v, q_num_head//kv_num_head)
    attention_score = torch.matmul(q, k.transpose(-1,-2))/math.sqrt(head_size)
    if mask:
        attention_score = attention_score.masked_fill(mask==0, -1e20)
    attention_score = F.softmax(attention_score, -1)
    attention_score = dropout(attention_score)
    
    output = torch.matmul(attention_score, v).permute(0, 2, 1, 3).contiguous()
    output = output.view(batch_size, seq_len, hidden_size)
    
    output = Wo(output)
    
    return output, kv_cache
    
batch_size, seq_len, hidden_size, num_head = 16, 20, 128, 5
position_ids = torch.arange(seq_len, dtype=torch.long)
X1 = torch.rand((batch_size, seq_len, hidden_size))
Y1, past_kv1 = multi_head_attention(X1, 64, 8)
X2 = torch.rand((batch_size, seq_len *2, hidden_size))
Y2, past_kv2 = multi_head_attention(X2, 64, 8, kv_cache=past_kv1)