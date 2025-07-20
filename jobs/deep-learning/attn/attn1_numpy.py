import numpy as np
import math
def softmax(x):
    expx = np.exp(x)
    return expx / np.sum(expx, axis=-1, keepdims=True)
def multi_head_attention(x, q_head_num):
    def transpose_for_score(x, head_num):
        new_shape = (x.shape[0], x.shape[1], head_num, x.shape[-1] // head_num)
        x = x.reshape(new_shape)
        return x.transpose(0,2,1,3)
    batch_size, seq_len, hidden_size = x.shape
    head_size = hidden_size // q_head_num
    Wq = np.random.randn(hidden_size, hidden_size)
    Wk = np.random.randn(hidden_size, hidden_size)
    Wv = np.random.randn(hidden_size, hidden_size)
    Wo = np.random.randn(hidden_size, hidden_size)
    
    query, keys, values = x@Wq, x@Wk, x@Wv
    query, keys, values = transpose_for_score(query, q_head_num), transpose_for_score(keys, q_head_num), transpose_for_score(values, q_head_num)
    scores = query@keys.transpose(0,1,3,2) / math.sqrt(hidden_size)
    scores = softmax(scores)
    scores = scores@values
    output = scores.transpose(0,2,1,3).reshape(batch_size, seq_len, hidden_size)
    output = output@Wo
    
    return output

X = np.random.rand(16, 20, 128)
Y = multi_head_attention(X, 64)
print(Y)