import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

# gqa + kv cache + rope
def sin_position_embedding(batch_size, num_head, seq_len, d):
    ids = torch.arange(0, d // 2)
    theta = torch.pow(10000, -2 * ids / d)
    position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(-1)
    embeddings = position_ids * theta
    # [seq len, d / 2]
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1) # [seq len, d/2, 2]
    embeddings = embeddings.repeat((batch_size, num_head, *([1] * len(embeddings.shape))))
    embeddings = embeddings.reshape(batch_size, num_head, seq_len, d)
    return embeddings
    
def rotary_emb(q, k):
    # [batch size, num heads, seq len, head size]
    batch_size, num_head_q, seq_len_q, d = q.shape
    _, num_head_k, seq_len_k, _ = k.shape
    pos_emb_q = sin_position_embedding(batch_size, num_head_q, seq_len_q, d)
    pos_emb_k = sin_position_embedding(batch_size, num_head_k, seq_len_k, d)
    # [batch size, num heads, seq len, d]
    cos_pos = pos_emb_q[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb_q[..., ::2].repeat_interleave(2, dim=-1)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
    q = q*cos_pos + q2*sin_pos
    
    cos_pos = pos_emb_k[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb_k[..., ::2].repeat_interleave(2, dim=-1)
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)
    k = k*cos_pos + k2*sin_pos
    import pdb;pdb.set_trace()
    return q, k
    
def group_query_attention(inputs, q_num_head, kv_num_head, past_kv=None, mask=None, ep=-1e20):
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
    
    queries, keys = rotary_emb(queries, keys)
    if past_kv is not None:
        keys = torch.cat([keys, past_kv[0]], dim=2)
        values = torch.cat([values, past_kv[1]], dim=2)
    past_kv = (keys, values)
    
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
    
    return outputs, past_kv

batch_size, seq_len, hidden_size, num_head = 16, 20, 128, 5
position_ids = torch.arange(seq_len, dtype=torch.long)
X1 = torch.rand((batch_size, seq_len, hidden_size))
Y1, past_kv1 = group_query_attention(X1, 64, 8)
X2 = torch.rand((batch_size, seq_len *2, hidden_size))
Y2, past_kv2 = group_query_attention(X2, 64, 8, past_kv=past_kv1)
