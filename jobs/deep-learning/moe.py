import torch
import torch.nn as nn
import torch.nn.functional as F

def token_level_moe(inputs, top_k=2, num_experts=4, output_dim=5):
    batch_size, seq_len, input_dim = inputs.shape
    experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
    gate = nn.Linear(input_dim, num_experts)
    num_experts = len(experts)

    flattened_inputs = inputs.view(-1, input_dim) #(bs, h)

    gate_logits = F.softmax(gate(flattened_inputs), dim=-1)  # (bs, n)

    weights, selected_experts = torch.topk(gate_logits, k=top_k, dim=-1)  # (bs, k)
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    import pdb;pdb.set_trace()
    output_dim = experts[0].out_features
    results = torch.zeros((batch_size * seq_len, output_dim), dtype=inputs.dtype, device=inputs.device)

    for i, expert in enumerate(experts):
        batch_idx, nth_expert = torch.where(selected_experts == i)
        if batch_idx.numel() > 0:
            expert_outputs = expert(flattened_inputs[batch_idx])
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_outputs

    return results.view(batch_size, seq_len, output_dim)


input_dim, hidden_dim, output_dim, num_experts, top_k = 10, 32, 5, 4, 2
batch_size, seq_len = 2, 3
x = torch.randn(batch_size, seq_len, input_dim)
output = token_level_moe(x, num_experts=num_experts, top_k=top_k, output_dim=output_dim)

