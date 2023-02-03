import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .STE import *

import torch
import torch.nn as nn


class SBMAttention(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        self.drop_attn = nn.Dropout(p=config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.num_clusters = config["num_clusters"][idx]
        self.dropout = nn.Dropout(0.2)
        self.layer = nn.Embedding(self.num_head * self.num_clusters, self.head_dim)
        self.orth_clusters = self.layer

        self.proj = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.head_dim, self.head_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.head_dim, self.head_dim),
        )

    def forward(self, Q, K, V, mask):

        b, h, n, d = Q.shape
        _, _, m, _ = V.shape
        k = self.num_clusters
        self.clusters = self.orth_clusters.weight.reshape(h, k, -1)
        dist = torch.matmul(self.clusters, torch.transpose(self.clusters, -1, -2))
        S = nn.Softmax(dim=-1)(dist.reshape(self.num_head, self.num_clusters**2)).reshape(self.num_head, k, k).unsqueeze(0).repeat((b, 1, 1, 1))

        Qhat = nn.Sigmoid()(
            torch.matmul(
                self.proj(Q),
                self.clusters.transpose(-1, -2),
            )
        )

        Khat = nn.Sigmoid()(
            torch.matmul(
                self.proj(K),
                self.clusters.transpose(-1, -2),
            )
        )
        # B, H, n, num_cluster
        expA = torch.matmul(Qhat, torch.matmul(S, Khat.transpose(-1, -2)))

        graph = SampleGraphSparseGraph.apply(expA)

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot.masked_fill_(mask[:, None, None, :] == 1, float("-inf"))  # first apply user-provided mask
        attn = F.normalize(nn.Softmax(dim=-1)(dot) * graph, p=1, dim=-1)
        X = torch.matmul(self.drop_attn(attn), V)  # apply dropout then matmul
        sparsity = torch.sum(graph, dim=(0, -1, -2)) / (b * n * m)  # head-wise sparsity

        return X, sparsity, graph, attn


class FullAttention(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        self.drop_attn = nn.Dropout(p=config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.dropout = nn.Dropout(0.2)

    def forward(self, Q, K, V, mask):

        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot.masked_fill_(mask[:, None, None, :] == 1, float("-inf"))  # first apply user-provided mask
        attn = F.normalize(nn.Softmax(dim=-1)(dot), p=1, dim=-1)
        X = torch.matmul(self.drop_attn(attn), V)  # apply dropout then matmul
        sparsity = None
        graph = mask

        return X, sparsity, graph, attn


class Attention(nn.Module):
    def __init__(self, config, idx, full_att=False):
        super().__init__()

        self.grad_checkpointing = config["attention_grad_checkpointing"]

        self.dim = config["transformer_dim"]
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.attn_type = config["attn_type"]

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        if full_att:
            self.attn = FullAttention(config, idx)
        else:
            self.attn = SBMAttention(config, idx)

        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)

    def forward(self, inputs):

        X, mask, deliver = inputs
        Q = self.split_heads(self.W_q(X))  # B, H, N, d
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        with torch.cuda.amp.autocast(enabled=False):
            attn_out, sparsity, graph, attn = self.attn(
                Q.float(),
                K.float(),
                V.float(),
                mask.float(),
            )

        attn_out = self.combine_heads(attn_out)  # B, N, d
        out = self.ff(attn_out)
        return out, sparsity, graph, attn

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
