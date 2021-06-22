import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import pickle

import math
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.vocab_size = 65
        self.block_size = block_size
        self.data = data
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

block_size = 128
tokens = pickle.load(open('tokens.pkl','rb'))
train_dataset = CharDataset(tokens, block_size)


from mingpt.model import GPT, GPTConfig
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
model = GPT(mconf)

from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample
while True:
    tconf = TrainerConfig(max_epochs=1, batch_size=256, learning_rate=6e-4)
    trainer = Trainer(model, train_dataset, tconf)
    trainer.train()