import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_sin_pe(x):
    batch_size, head_num, seq_len, head_size = x.shape
    d_id = torch.arange(0, head_size // 2)
    d_id = torch.pow(10000, -2*d_id/head_size)
    s_id = torch.arange(0, seq_len).unsqueeze(-1)
    pe = s_id * d_id
    pe = torch.stack([torch.sin(pe), torch.cos(pe)], dim=-1)
    pe = pe.repeat((batch_size, head_num, 1, 1, 1))
    return pe.reshape(batch_size, head_num, seq_len, head_size)
def rope(x):
    pe = get_sin_pe(x)
    cos_pe = pe[..., 1::2].repeat_interleave(2, -1)
    sin_pe = pe[..., ::2].repeat_interleave(2, -1)
    x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape)
    return x*cos_pe+x2*sin_pe
    
def multi_head_latent_attention(x, q_lora_rank, kv_lora_rank, q_num_head, q_head_size, qk_nope_dim, qk_rope_dim, v_head_size, past_kv=None, mask=None):
    batch_size, seq_len, hidden_size = x.shape
    q_down = nn.Linear(hidden_size, q_lora_rank)
    q_up = nn.Linear(q_lora_rank, q_num_head * q_head_size)
    q_norm = nn.RMSNorm(q_lora_rank)
    
    kv_down = nn.Linear(hidden_size, kv_lora_rank + qk_rope_dim)
    kv_up = nn.Linear(kv_lora_rank, q_num_head * (qk_nope_dim + v_head_size))
    kv_norm = nn.RMSNorm(kv_lora_rank)
    
    q = q_up(q_norm(q_down(x)))
    q = q.view(batch_size, seq_len, q_num_head, q_head_size).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [qk_nope_dim, qk_rope_dim], -1)
    
    kv = kv_down(x)
    kv, k_pe = torch.split(kv, [kv_lora_rank, qk_rope_dim], dim=-1)
    k_pe = k_pe.view(batch_size, seq_len, 1, qk_rope_dim).transpose(1, 2)
    kv = kv_up(kv_norm(kv))
    kv = kv.view(batch_size, seq_len, q_num_head, qk_nope_dim + v_head_size).transpose(1, 2)
    k_nope, v = torch.split(kv, [qk_nope_dim, v_head_size], -1)
    
    q_pe, k_pe = rope(q_pe), rope(k_pe)
    
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, q_num_head, -1, -1)], dim=-1)
    
    if past_kv:
        k = torch.cat([k, past_kv[0]], dim=2)
        v = torch.cat([v, past_kv[1]], dim=2)
    past_kv = (k, v)
    attention_score = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(q_head_size)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask==0, -1e20)
    attention_score = F.softmax(attention_score, -1)
    
    output = torch.matmul(attention_score, v).permute(0, 2, 1, 3).contiguous()
    output = output.view(batch_size, seq_len, hidden_size)
    return output, past_kv

batch_size, seq_len, hidden_size, num_head = 16, 20, 128, 5
X1 = torch.rand((batch_size, seq_len, hidden_size,))
Y1, past_kv1 = multi_head_latent_attention(X1, 2, 4, 8, 10, 8, 2, 10)