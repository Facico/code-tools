import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from copy import deepcopy

policy_model = LlamaForCausalLM(config=LlamaConfig(vocab_size=1000, num_hidden_layers=1, hidden_size=128))
ref_model = deepcopy(policy_model)
input_prompt = [1, 2, 3]
good_response = [4, 5]
bad_response = [6, 7]
inputs_ids = torch.LongTensor([input_prompt + good_response, input_prompt + bad_response])
labels = torch.LongTensor([[-100] * len(input_prompt) + good_response, [-100] * len(input_prompt) + bad_response])
labels = labels[:,1:]
mask = (labels!=-100)
labels[labels == -100] = 0
beta = 0.01

logits = policy_model(inputs_ids)["logits"][:,:-1,:] #B S V
logits = torch.gather(logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) #B S
logits = (logits*mask).sum(-1) # B
good_logits, bad_logits = logits[0], logits[1:]

with torch.no_grad():
    logits = ref_model(inputs_ids)["logits"][:,:-1,:] #B S V
    logits = torch.gather(logits.log_softmax(-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1) #B S
    logits = (logits*mask).sum(-1) # B
    ref_good_logits, ref_bad_logits = logits[0], logits[1:]
dpo_logits = (good_logits-ref_good_logits) - (bad_logits - ref_bad_logits)
loss = -F.logsigmoid(beta*dpo_logits).mean()
print(loss)