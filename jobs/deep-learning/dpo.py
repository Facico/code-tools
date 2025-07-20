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

logits = policy_model(inputs_ids)["logits"][:,:-1,:]
token_p = torch.gather(logits.log_softmax(-1),dim=2,index=labels.unsqueeze(2)).squeeze(2)
all_p = (token_p * mask).sum(-1)
good_p, bad_p = all_p[0], all_p[1:]

with torch.no_grad():
    logits = ref_model(inputs_ids)["logits"][:,:-1,:]
    token_p = torch.gather(logits.log_softmax(-1,),dim=2,index=labels.unsqueeze(2)).squeeze(2)
    all_p = (token_p * mask).sum(-1)
    ref_good_p, ref_bad_p = all_p[0], all_p[1:]
log_logits = (good_p - ref_good_p) - (bad_p - ref_bad_p)
loss = -F.logsigmoid(beta*log_logits).mean()
print(loss) #-0.6931