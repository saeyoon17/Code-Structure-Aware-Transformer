import copy
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn

__all__ = [
    "_get_clones",
    "FeedForward",
    "Embeddings",
    "SublayerConnection",
    "transpose_for_scores",
    "DecoderLayer",
    "BaseDecoder",
    "Generator",
]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Embeddings(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout=0.1, with_pos=False):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if with_pos:
            self.pos_emb = PositionalEncoding(hidden_size)
        else:
            self.pos_emb = None
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        words_embeddings = self.word_embeddings(x)
        if self.pos_emb is not None:
            words_embeddings = self.pos_emb(words_embeddings)

        embeddings = self.norm(words_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * -(math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feed_forward, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feed_forward)
        self.linear2 = nn.Linear(dim_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x)))), None


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        output, attn_weights = sublayer(self.norm(x))
        return x + self.dropout(output), attn_weights


def transpose_for_scores(x, num_heads):
    new_x_shape = x.size()[:-1] + (num_heads, -1)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


class Generator(nn.Module):
    def __init__(self, tgt_vocab_size, hidden_size, dropout):
        super(Generator, self).__init__()
        self.soft_max = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, outputs):
        out = self.linear(outputs)
        gen_prob = self.soft_max(self.dropout(out))
        return torch.log(gen_prob)


class BaseDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(BaseDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt,
        memory,
        tgt_mask,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        output = tgt

        for mod in self.layers:
            output, attn_weights = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_weights


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout=dropout)
        self.sublayer = _get_clones(SublayerConnection(d_model, dropout), 3)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(DecoderLayer, self).__setstate__(state)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        tgt, attn_weights = self.sublayer[0](
            tgt,
            lambda x: self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask),
        )

        tgt, attn_weights = self.sublayer[1](
            tgt,
            lambda x: self.multihead_attn(
                x,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            ),
        )

        tgt, _ = self.sublayer[2](tgt, self.feed_forward)
        return tgt, attn_weights
