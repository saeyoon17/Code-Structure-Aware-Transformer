import torch.utils.data
import torch
import torch.nn as nn

from module import (
    Embeddings,
    DisentangledAttn,
    FeedForward,
    SublayerConnection,
    _get_clones,
)
from module.base_seq2seq import BaseTrans
from module.components import BaseDecoder, DecoderLayer, Generator
from module.sbm_model import SBM

__all__ = ["CSATrans"]


class TreePositionalEncodings(torch.nn.Module):
    # Novel positional encodings to enable tree-based transformers
    # https://papers.nips.cc/paper/2019/file/6e0917469214d8fbd8c517dcdc6b8dcf-Paper.pdf
    def __init__(self, depth, degree, n_feat, d_model):
        """
        depth: max tree depth
        degree: max num children
        n_feat: number of features
        d_model: size of model embeddings
        """
        super(TreePositionalEncodings, self).__init__()
        self.depth = depth
        self.width = degree
        self.d_pos = n_feat * depth * degree
        self.d_model = d_model
        self.d_tree_param = self.d_pos // (self.depth * self.width)
        self.p = torch.nn.Parameter(torch.ones(self.d_tree_param, dtype=torch.float32), requires_grad=True)
        self.init_weights()

    def init_weights(self):
        self.p.data.uniform_(0.7, 0.999)

    def build_weights(self):
        d_tree_param = self.d_tree_param
        tree_params = torch.tanh(self.p)
        tiled_tree_params = tree_params.reshape((1, 1, -1)).repeat(self.depth, self.width, 1)
        tiled_depths = torch.arange(self.depth, dtype=torch.float32, device=self.p.device).reshape(-1, 1, 1).repeat(1, self.width, d_tree_param)
        tree_norm = torch.sqrt((1 - torch.square(tree_params)) * self.d_model / 2)
        tree_weights = (torch.pow(tiled_tree_params, tiled_depths) * tree_norm).reshape(self.depth * self.width, d_tree_param)
        return tree_weights

    def treeify_positions(self, positions, tree_weights):
        treeified = positions.unsqueeze(-1) * tree_weights
        shape = treeified.shape
        shape = shape[:-2] + (self.d_pos,)
        treeified = torch.reshape(treeified, shape)
        return treeified

    def forward(self, positions):
        """
        positions: Tensor [bs, n, width * depth]
        returns: Tensor [bs, n, width * depth * n_features]
        """
        tree_weights = self.build_weights()
        positions = self.treeify_positions(positions, tree_weights)
        return positions


class CSATrans(BaseTrans):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        hidden_size,
        num_heads,
        num_layers,
        sbm_layers,
        use_pegen,
        dim_feed_forward,
        dropout,
        pe_dim,
        pegen_dim,
        sbm_enc_dim,
        clusters,
        full_att,
        state_dict=None,
        max_src_len=150,
    ):

        super(CSATrans, self).__init__()
        self.num_heads = num_heads
        self.pe_dim = pe_dim
        self.pegen_dim = pegen_dim
        sbm_enc_dim = sbm_enc_dim
        self.src_embedding = Embeddings(
            hidden_size=sbm_enc_dim - self.pe_dim,
            vocab_size=src_vocab_size,
            dropout=dropout,
            with_pos=False,
        )

        self.tgt_embedding = Embeddings(
            hidden_size=hidden_size,
            vocab_size=tgt_vocab_size,
            dropout=dropout,
            with_pos=True,
        )
        self.use_pegen = use_pegen
        if self.use_pegen == "pegen":
            self.src_pe_embedding = Embeddings(
                hidden_size=self.pegen_dim,
                vocab_size=src_vocab_size,
                dropout=dropout,
                with_pos=False,
            )
            encoder_layer1 = CSE_layer(
                self.pegen_dim,
                self.num_heads,
                self.pegen_dim,
                dropout,
            )
            self.pegen = CSE(
                encoder_layer1,
                num_layers,
                num_heads,
                self.pegen_dim,
                dropout=dropout,
                max_src_len=max_src_len,
            )
        elif self.use_pegen == "laplacian":
            pass
        elif self.use_pegen == "sequential":
            pass
        elif self.use_pegen == "treepos":
            self.tree_pos_enc = TreePositionalEncodings(
                depth=16,
                degree=8,
                n_feat=self.pegen_dim // 128,
                d_model=self.pegen_dim,
            )
        elif self.use_pegen == "triplet":
            # python
            self.triplet_emb = nn.Embedding(1246, self.pegen_dim)
            # java
            # self.triplet_emb = nn.Embedding(1505, self.pegen_dim)
        config = {
            "sbm_layers": sbm_layers,
            "transformer_dim": sbm_enc_dim,
            "transformer_hidden_dim": sbm_enc_dim,
            "head_dim": sbm_enc_dim // num_heads,
            "num_head": num_heads,
            "attn_type": "sbm",
            "attention_grad_checkpointing": False,
            "attention_dropout": 0.2,
            "num_clusters": clusters,
            "dropout_prob": 0.2,
            "out_dim": hidden_size,
            "max_src_len": max_src_len,
            "full_att": full_att,
        }
        self.SBM = SBM(config, sbm_enc_dim, self.pe_dim, self.pegen_dim, self.use_pegen)
        decoder_layer = DecoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout, activation="gelu")
        self.decoder = BaseDecoder(decoder_layer, 4, norm=nn.LayerNorm(hidden_size))
        self.generator = Generator(tgt_vocab_size, hidden_size, dropout)

        print("Init or load model.")
        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            try:
                torch.nn.init.orthogonal_(self.SBM.transformer_0.mha.attn.layer.weight)
                torch.nn.init.orthogonal_(self.SBM.transformer_1.mha.attn.layer.weight)
                torch.nn.init.orthogonal_(self.SBM.transformer_2.mha.attn.layer.weight)
                torch.nn.init.orthogonal_(self.SBM.transformer_3.mha.attn.layer.weight)
            except:
                pass
        else:
            self.load_state_dict(state_dict)


class CSE(nn.Module):
    def __init__(self, encoder_layer, num_layers, num_heads, hidden_size, dropout=0.2, max_src_len=150):
        super(CSE, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_heads = num_heads
        d_k = hidden_size // (num_heads)
        self.hidden_size = hidden_size
        self.d_k = d_k
        self.max_src_len = max_src_len
        self.edge_dim = d_k
        self.L_q = nn.Embedding(max_src_len, hidden_size)  # rel, d
        self.T_q = nn.Embedding(max_src_len, hidden_size)  # rel, d
        self.f = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def build_rel_emb(self):
        L_q = self.L_q.weight
        T_q = self.T_q.weight

        rel_q = torch.stack([L_q, T_q])  # 2, 150, d

        return [rel_q]

    def forward(self, data):
        output = data.src_pe_emb
        L = data.L.unsqueeze(1).repeat(1, 4, 1, 1)
        T = data.T.unsqueeze(1).repeat(1, 4, 1, 1)
        rel = torch.cat([L, T], dim=1).to(torch.int64)  # B, 8, N, N
        L_mask = data.L_mask.unsqueeze(1).repeat(1, 4, 1, 1)
        T_mask = data.T_mask.unsqueeze(1).repeat(1, 4, 1, 1)
        mask = torch.cat([L_mask, T_mask], dim=1)
        rel_emb = self.build_rel_emb()  # B, H, L, d

        for i, layer in enumerate(self.layers):
            output = layer(output, rel_emb, rel, mask)

        return self.norm(output)


class CSE_layer(nn.Module):
    def __init__(self, hidden_size, num_heads, dim_feed_forward, dropout):
        super(CSE_layer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.self_attn = DisentangledAttn(num_heads, hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, dim_feed_forward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.sublayer = _get_clones(SublayerConnection(hidden_size, dropout), 2)

    def forward(self, src, rel_emb, rel, mask):
        src, attn_weights = self.sublayer[0](
            src,
            lambda x: self.self_attn(x, x, x, rel_emb, rel, mask),
        )
        src, _ = self.sublayer[1](src, self.feed_forward)
        return src
