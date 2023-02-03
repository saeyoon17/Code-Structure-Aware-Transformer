import time
import torch
import torch.nn as nn
from dataset import make_std_mask
from torch.autograd import Variable
from utils import BOS, PAD, UNK
import numpy as np

__all__ = ["BaseTrans", "GreedyGenerator"]


def eig(sym_mat):
    # (sorted) eigenvectors with numpy
    EigVal, EigVec = np.linalg.eigh(sym_mat)

    # for eigval, take abs because numpy sometimes computes the first eigenvalue approaching 0 from the negative
    eigvec = torch.from_numpy(EigVec).float()  # [N, N (channels)]
    eigval = torch.from_numpy(np.sort(np.abs(np.real(EigVal)))).float()  # [N (channels),]
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


def lap_eig(dense_adj, number_of_nodes, in_degree):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    https://github.com/DevinKreuzer/SAN/blob/main/data/molecules.py
    """
    dense_adj = dense_adj.detach().float().cpu().numpy()
    in_degree = in_degree.detach().float().cpu().numpy()

    # Laplacian
    A = dense_adj
    N = np.diag(in_degree.clip(1) ** -0.5)
    L = np.eye(number_of_nodes) - N @ A @ N

    eigvec, eigval = eig(L)
    return eigvec, eigval  # [N, N (channels)]  [N (channels),]


class BaseTrans(nn.Module):
    def __init__(self):
        super(BaseTrans, self).__init__()

    def base_process(self, data):
        # t = time.time()
        src_seq = data.src_seq
        data.src_mask = src_seq.eq(PAD)
        data.src_emb = self.src_embedding(src_seq)
        if self.use_pegen == "pegen":
            data.src_pe_emb = self.src_pe_embedding(src_seq)

        if data.tgt_seq is not None:
            tgt_seq = data.tgt_seq
            data.tgt_mask = make_std_mask(tgt_seq, PAD)
            data.tgt_emb = self.tgt_embedding(tgt_seq)

    def process_data(self, data):
        self.base_process(data)

    def forward(self, data):
        t1 = time.time()
        self.process_data(data)
        encoder_outputs, sparsity, src_pe, graphs, attns = self.encode(data)
        decoder_outputs, attn_weights = self.decode(data, encoder_outputs)
        out = self.generator(decoder_outputs)
        return out, sparsity, src_pe, graphs, attns

    def encode(self, data):
        if self.use_pegen == "pegen":
            src_pe = self.pegen(data)
        elif self.use_pegen == "laplacian":
            B = data.src_emb.size(0)
            N = data.src_emb.size(1)
            src_pes = []
            for i in range(B):
                num_node = data.num_node[i]
                adj = data.adj[i][:num_node, :num_node]  # N, N
                in_degree = adj.long().sum(dim=1).view(-1)
                EigVec, EigVal = lap_eig(adj, num_node, in_degree)  # EigVec: [N, N]
                src_pe = torch.zeros(N, self.pegen_dim).to("cuda")  # [N, half_pos_enc_dim]
                src_pe[:num_node, :num_node] = EigVec
                src_pes.append(src_pe)
            src_pe = torch.stack(src_pes)
        elif self.use_pegen == "treepos":
            src_pe = self.tree_pos_enc(data.tree_pos)
        elif self.use_pegen == "sequential":
            src_pe = None
        elif self.use_pegen == "triplet":
            src_pe = self.triplet_emb(data.triplet)

        tmp, sparsity, graphs, attns, pe = self.SBM(data, src_pe, self.use_pegen)

        if sparsity == (None, None, None, None):
            sparsity = 1
        else:
            sparsity = torch.mean(torch.stack((sparsity)))

        return tmp, sparsity, pe, graphs, attns

    def decode(self, data, encoder_outputs):
        tgt_emb = data.tgt_emb
        tgt_mask = data.tgt_mask
        src_mask = data.src_mask

        tgt_emb = tgt_emb.permute(1, 0, 2)  #
        encoder_outputs = encoder_outputs.permute(1, 0, 2)  #
        tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)  #
        outputs, attn_weights = self.decoder(
            tgt=tgt_emb,
            memory=encoder_outputs,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
        )
        outputs = outputs.permute(1, 0, 2)
        return outputs, attn_weights


class GreedyGenerator(nn.Module):
    def __init__(self, model, max_tgt_len, multi_gpu=False):
        super(GreedyGenerator, self).__init__()
        if multi_gpu:
            self.model = model.module
        else:
            self.model = model
        self.max_tgt_len = max_tgt_len
        self.start_pos = BOS
        self.unk_pos = UNK

    def forward(self, data):
        # No need to do so it will be replaced.
        # data.tgt_seq = None
        self.model.process_data(data)

        encoder_outputs, sparsity, src_pe, graphs, attns = self.model.encode(data)
        batch_size = encoder_outputs.size(0)
        ys = torch.ones(batch_size, 1).fill_(self.start_pos).long().to(encoder_outputs.device)
        for i in range(self.max_tgt_len - 1):
            data.tgt_mask = make_std_mask(ys, 0)
            data.tgt_emb = self.model.tgt_embedding(Variable(ys))
            decoder_outputs, decoder_attn = self.model.decode(data, encoder_outputs)
            out = self.model.generator(decoder_outputs)
            out = out[:, -1, :]
            _, next_word = torch.max(out, dim=1)
            ys = torch.cat([ys, next_word.unsqueeze(1).long().to(encoder_outputs.device)], dim=1)

        return ys[:, 1:]
