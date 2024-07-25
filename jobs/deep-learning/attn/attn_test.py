import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def multi_head_attn(x, q_head_num, kv_head_num, past_kv=None):
    def transpose_for_score(x, head_num):
        new_shape = (x.shape[0], x.shape[1], head_num, x.shape[-1] // head_num)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    def repeat_kv(x, repeat_num):
        # x : [batch size, head num, seq len, head size] => [batch size, head num * repeat num, seq len, head size]
        if repeat_num == 1:
            return x
        batch_size, head_num, seq_len, head_size = x.shape
        x = x[:, :, None, :, :].expand(batch_size, head_num, repeat_num, seq_len, head_size)
        return x.reshape(batch_size, head_num*repeat_num, seq_len, head_size)
    # x : [batch size, seq len, hidden size]
    batch_size, seq_len, hidden_size = x.shape
    head_size = hidden_size // q_head_num
    Wq = nn.Linear(hidden_size, head_size * q_head_num)
    Wk = nn.Linear(hidden_size, head_size * kv_head_num)
    Wv = nn.Linear(hidden_size, head_size * kv_head_num)
    Wo = nn.Linear(hidden_size, hidden_size)
    
    query_state = transpose_for_score(Wq(x), q_head_num)
    key_state = transpose_for_score(Wk(x), kv_head_num)
    value_state = transpose_for_score(Wv(x), kv_head_num)
    
    if past_kv is not None:
        key_state = torch.cat((key_state, past_kv[0]), dim=2)
        value_state = torch.cat((value_state, past_kv[1]), dim=2)
    past_kv = (key_state, value_state)
    
    key_state = repeat_kv(key_state, q_head_num // kv_head_num)
    value_state = repeat_kv(value_state, q_head_num // kv_head_num)
    
    attention_score = torch.matmul(query_state, value_state.transpose(-1, -2))
    attention_score /= math.sqrt(head_size)
    attention_score =  F.softmax(attention_score, dim=-1)
    attention_weight = torch.matmul(attention_score, value_state)
    attention_weight = attention_weight.permute(0, 2, 1, 3).contiguous()
    attention_weight = attention_weight.view(batch_size, seq_len, hidden_size)
    
    output = Wo(attention_weight)
    return output, past_kv

batch_size, seq_len, hidden_size, num_head = 16, 20, 128, 5
X1 = torch.rand((batch_size, seq_len, hidden_size))
Y1, past_kv1 = multi_head_attn(X1, 64, 8)
X2 = torch.rand((batch_size, seq_len, hidden_size))
Y2, past_kv2 = multi_head_attn(X2, 64, 8, past_kv=past_kv1)