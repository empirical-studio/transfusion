#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

import yaml

with open('config.yml', 'r') as f:
  config = yaml.safe_load(f)
  for k, v in config.items():
    globals()[k] = v
  # print(f"{config = }")

# # hyperparameters
# batch_size        = config.get("batch_size") # how many independent sequences will we process in parallel?
# block_size        = config.get("block_size") # what is the maximum context length for predictions?
# max_iters         = config.get("max_iters")
# eval_interval     = config.get("eval_interval")
# learning_rate     = config.get("learning_rate")
# eval_iters        = config.get("eval_iters")
# embeddings_size   = config.get("embeddings_size")
# head_count        = config.get("head_count")
# n_layer           = config.get("n_layer")
# dropout           = config.get("dropout")
# seed              = config.get("seed")
# # ------------

torch.manual_seed(seed)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# with open('data/input.txt', 'r', encoding='utf-8') as f:
#   text = f.read()

import datasets

# # load tiny shakespeare dataset
# # dataset = datasets.load_dataset('tiny_shakespeare', cache_dir="cache")

dataset = datasets.load_dataset('siavava/ai-tech-articles', split='train')
df = dataset.to_pandas()
df = df[df["year"] == 2023]

# concat all "text" column values into one string
text = " ".join(df["text"].tolist())

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
  """
    Generate a small batch of data of inputs (`x`) and targets (`y`).

    Note; while explicit access to the data is not provided,
    the data is closed over in the function scope.
  """
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

# exported functions
__all__ = ["get_batch", "encode", "decode"]
