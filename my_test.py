from torch import nn
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import AutoTokenizer  
from math import sqrt

model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "time flies like an arrow"
inputs = tokenizer(text, return_tensors='pt',add_special_tokens=False)
print(inputs)
print(inputs.input_ids)

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
print(config)
print(token_emb)

inputs_embeds = token_emb(inputs.input_ids)
print(inputs_embeds)
print(inputs_embeds.size())

Q = K = V = inputs_embeds
dim_k = K.size(-1) 
print(K.size(-2))
scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(dim_k)
print(scores)
print(scores.size())

weights = F.softmax(scores, dim=-1)
print(weights)
print(weights.sum(dim=-1))

attn_outputs = torch.bmm(weights, V)
print(attn_outputs)
print(attn_outputs.shape)

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)