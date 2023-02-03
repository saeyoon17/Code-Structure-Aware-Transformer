import torch
import torch.nn as nn
import torch.nn.functional as F
from module.components import PositionalEncoding
from .sbm_attn import Attention

__all__ = ["Model"]


class Transformer(nn.Module):
    def __init__(self, config, idx):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])
        self.mha = Attention(config, idx, config["full_att"])
        self.dropout1 = torch.nn.Dropout(p=config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
            nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
            nn.GELU(),
            torch.nn.Dropout(p=config["dropout_prob"]),
            nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
            torch.nn.Dropout(p=config["dropout_prob"]),
        )

    def forward(self, X, mask, deliver):
        out, sparsity, graph, attn = self.mha([self.norm1(X), mask, deliver])
        X = self.dropout1(out) + X
        X = self.mlpblock(self.norm2(X)) + X
        return X, sparsity, graph, attn


class SBM(nn.Module):
    def __init__(self, config, sbm_enc_dim, pe_dim, pegen_dim, use_pegen):
        super().__init__()

        self.num_layers = config["sbm_layers"]
        self.tied_weights = False
        self.attn_type = "sbm"
        for idx in range(self.num_layers):
            setattr(self, f"transformer_{idx}", Transformer(config, idx))
        self.norm = nn.LayerNorm(sbm_enc_dim)
        self.out = nn.Linear(sbm_enc_dim, config["out_dim"])
        if use_pegen == "sequential":
            self.pe = PositionalEncoding(sbm_enc_dim, config["max_src_len"])
        else:
            self.pe_expand = nn.Linear(pegen_dim, pe_dim)

    def forward(self, data, src_pe, use_pe):

        mask = data.src_mask
        if use_pe != "sequential":
            pe = self.pe_expand(src_pe)
            X = torch.cat([data.src_emb, pe], dim=-1)
        else:
            pe = None
            X = self.pe(data.src_emb)
        deliver = []
        all_sparsity = ()
        graphs = []  # attention masks per layer
        attns = []  # attention scores per layer
        for idx in range(self.num_layers):
            X, sparsity, graph, attn = getattr(self, f"transformer_{idx}")(X, mask, deliver)
            all_sparsity += (sparsity,)
            graphs.append(graph)
            attns.append(attn)
        X = self.norm(X) * ~mask[:, :, None]
        X = self.out(X)
        return X, all_sparsity, graphs, attns, pe
