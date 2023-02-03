import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from module import _get_clones, transpose_for_scores

__all__ = ["DisentangledAttn"]


class DisentangledAttn(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(DisentangledAttn, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = _get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.l_linear = _get_clones(nn.Linear(d_model, self.d_k * 4), 2)
        self.t_linear = _get_clones(nn.Linear(d_model, self.d_k * 4), 2)

    def forward(self, query, key, value, rel_emb, rel, mask):

        query, key, value = [transpose_for_scores(l(x), self.h) for l, x in zip(self.linear_layers, (query, key, value))]
        lq = rel_emb[0]
        l = lq[0].unsqueeze(0)  # 1, L, d
        t = lq[1].unsqueeze(0)  # 1, L, d
        lq, lk = [transpose_for_scores(l(x), 4) for l, x in zip(self.l_linear, (l, l))]  # 1, 1, 4, L, d
        tq, tk = [transpose_for_scores(l(x), 4) for l, x in zip(self.t_linear, (t, t))]  # 1, 1, 4, L, d

        lq = torch.cat([lq, tq], dim=1)  # 1, 1, 8, L, d
        lk = torch.cat([lk, tk], dim=1)

        output = self.rel_attn(query, key, value, lq, lk, rel, mask)
        # B, H, seq_len, d

        output = output.permute(0, 2, 1, 3).contiguous()  # B, seq_len, H, d
        new_value_shape = output.size()[:-2] + (-1,)  # B, seq_len, d
        output = output.view(*new_value_shape)
        output = self.linear_layers[-1](output)
        return output, None

    @staticmethod
    def rel_attn(q, k, v, lq, lk, rel, mask):
        B, H, N, d_k = q.size()

        L = lq.size(2)
        scale = math.sqrt(d_k * 3)

        c2c = q @ k.permute(0, 1, 3, 2)  # B, H, N, N
        c2c = c2c / scale

        p2c = lq @ k.permute(0, 1, 3, 2)  # B, H, L, N
        rel_ = rel.transpose(-2, -1)  # b, h, j, i
        p2c = torch.gather(p2c, 2, rel_) / scale

        c2p = q @ lk.permute(0, 1, 3, 2)  # B, H, N, L
        c2p = torch.gather(c2p, 3, rel) / scale

        att_score = c2c + p2c + c2p
        att_score = att_score.masked_fill(mask == 1, -1e9)  # B, 8, N, N
        att_score = F.softmax(att_score, dim=-1)  # B, H, N, N
        output = att_score @ v
        return output
