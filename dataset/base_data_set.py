import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data.dataset import T_co
from torch_geometric.data import Data
from tqdm import tqdm
from utils import PAD, UNK


__all__ = ["BaseASTDataSet", "subsequent_mask", "make_std_mask"]


class BaseASTDataSet(data.Dataset):
    def __init__(self, config, data_set_name):
        super(BaseASTDataSet, self).__init__()
        self.src_vocab = config.src_vocab
        self.tgt_vocab = config.tgt_vocab

    def collect_fn(self, batch):
        src_seq = []
        tgt_seq = []
        target = []
        L = []
        T = []
        L_masks = []
        T_masks = []
        num_node = []
        adjs = []
        tree_pos = []
        triplets = []
        for data, _ in batch:
            L_masks.append(data["L"].eq(0))
            T_masks.append(data["T"].eq(0))  # N, N
            L.append(torch.clamp(data["L"] + 75, min=0, max=149))
            T.append(torch.clamp(data["T"] + 75, min=0, max=149))
            src_seq.append(data["src_seq"])
            tgt_seq.append(data["tgt_seq"])
            target.append(data["target"])
            num_node.append(data["num_node"])
            adjs.append(data["adj"])
            tree_pos_i = data["tree_pos"]  # N, d
            N, d = tree_pos_i.size()
            tree_pos_i = torch.cat([tree_pos_i, torch.zeros((150 - N), d)], dim=0)
            tree_pos.append(tree_pos_i)
            triplets.append(data["triplet"])

        src_seq = torch.stack(src_seq, dim=0)
        tgt_seq = torch.stack(tgt_seq, dim=0)
        target = torch.stack(target, dim=0)
        L = torch.stack(L, dim=0)
        T = torch.stack(T, dim=0)
        L_mask = torch.stack(L_masks)
        T_mask = torch.stack(T_masks)
        num_node = torch.tensor(num_node)
        adj = torch.stack(adjs)
        tree_pos = torch.stack(tree_pos)
        triplet = torch.stack(triplets)

        return (
            Data(
                src_seq=src_seq,
                tgt_seq=tgt_seq,
                target=target,
                L=L,
                T=T,
                L_mask=L_mask,
                T_mask=T_mask,
                num_node=num_node,
                adj=adj,
                tree_pos=tree_pos,
                triplet=triplet,
            ),
            target,
        )

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index) -> T_co:
        pass

    def convert_ast_to_tensor(self, ast_seq):
        ast_seq = ast_seq[: self.max_src_len]
        ast_seq = [":".join(e.split(":")[1:-1]) for e in ast_seq]
        return word2tensor(ast_seq, self.max_src_len, self.src_vocab)

    def convert_nl_to_tensor(self, nl):
        nl = nl[: self.max_tgt_len - 2]
        nl = ["<s>"] + nl + ["</s>"]
        return word2tensor(nl, self.max_tgt_len, self.tgt_vocab)


def word2tensor(seq, max_seq_len, vocab):
    seq_vec = [vocab.w2i[x] if x in vocab.w2i else UNK for x in seq]
    seq_vec = seq_vec + [PAD for i in range(max_seq_len - len(seq_vec))]
    seq_vec = torch.tensor(seq_vec, dtype=torch.long)
    return seq_vec


def load_list(file_path):
    _data = []
    print(f"loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            _data.append(eval(line))
    return _data


def load_seq(file_path):
    data_ = []
    print(f"loading {file_path} ...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            data_.append(line.split())
    return data_


def load_matrices(file_path):
    print("loading matrices...")
    matrices = np.load(file_path, allow_pickle=True)
    return matrices


def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_sequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(sub_sequent_mask) != 0


def make_std_mask(nl, pad):
    "Create a mask to hide padding and future words."
    nl_mask = (nl == pad).unsqueeze(-2)
    nl_mask = nl_mask | Variable(subsequent_mask(nl.size(-1)).type_as(nl_mask.data))
    return nl_mask
