import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)          # (batch, seq_len, embed_dim)
        pooled = emb.mean(dim=1)         # среднее по sequence
        h = F.relu(self.fc1(pooled))
        out = self.fc2(h)
        return out
