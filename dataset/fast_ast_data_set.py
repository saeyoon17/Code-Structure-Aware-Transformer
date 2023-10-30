from collections import OrderedDict
from os.path import exists
import numpy as np
import ipdb
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from utils import UNK, Vocab, PAD
from dataset import BaseASTDataSet

__all__ = ["FastASTDataSet"]


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


def update_node_child_idx(root_node):
    for idx, e in enumerate(root_node.children):
        if e.label.split(":")[0] == "idx":
            e.child_idx = -1
        else:
            e.child_idx = idx
        update_node_child_idx(e)


def get_node_triplet(root_node):
    for idx, e in enumerate(root_node.children):
        e.node_triplet = str((e.level, e.parent.child_idx, e.child_idx))
        get_node_triplet(e)


class FastASTDataSet(BaseASTDataSet):
    def __init__(self, config, data_set_name):
        print("Data Set Name : < Fast AST Data Set >")
        super(FastASTDataSet, self).__init__(config, data_set_name)
        self.config = config
        self.max_src_len = config.max_src_len
        self.max_tgt_len = config.max_tgt_len
        self.max_src_len = config.max_src_len
        self.data_set_name = data_set_name
        print("loading " + data_set_name + " data...")
        data_dir = config.data_dir + "/" + data_set_name + "/"
        self.out_data_path = data_dir + "processed_data"

        if exists(f"{self.out_data_path}.pt"):
            print("loading existing dataset")
        else:
            ast_path = data_dir + "split_pot.seq"
            matrices_path = data_dir + "split_matrices.npz"

            self.ast_data = load_list(ast_path)
            self.nl_data = load_seq(data_dir + "nl.original")
            self.matrices_data = load_matrices(matrices_path)

            print("building dataset")
            self.edges_data = self.convert_ast_to_edges(config.max_src_len)
        self.edges_data = []
        split_path = f"{self.out_data_path}.pt"
        self.edges_data += torch.load(split_path)
        self.data_set_len = len(self.edges_data)
        print(f"dataset lenght: {self.data_set_len}")

    def gen_tree_positions(self, rfs, width, height):
        position_dict = OrderedDict()
        init_tree_pos = torch.tensor([])

        for idx, e in enumerate(rfs):
            if idx == 0:
                tree_pos = init_tree_pos
                position_dict[e.num] = tree_pos
                continue
            level = min(e.level, height - 1)
            child_idx = min(e.child_idx, width - 1)
            try:
                # inherit from parent
                tree_pos = position_dict[e.parent.num].clone()
                tmp_pos = torch.zeros(width)
                tmp_pos[child_idx] = 1
                tree_pos = torch.cat([tmp_pos, tree_pos])
                position_dict[e.num] = tree_pos
            except:
                ipdb.set_trace()
        return position_dict

    def convert_ast_to_edges(self, max_src_len):
        print("building edges.")
        root_first_seqs = self.matrices_data["root_first_seq"]
        Ts = self.matrices_data["T"]
        Ls = self.matrices_data["L"]

        edges_data = []
        for i in tqdm(range(len(root_first_seqs))):
            root_first_seq = root_first_seqs[i][:max_src_len]

            pos_vocab = Vocab(need_bos=False, file_path="node_triplet_dictionary_java.pt")
            pos_vocab.load()
            root_first_seq[0].child_idx = 0
            root_first_seq[0].node_triplet = str((0, 0, 0))
            update_node_child_idx(root_first_seq[0])
            get_node_triplet(root_first_seq[0])
            triplet = torch.tensor([pos_vocab.w2i[e.node_triplet] if e.node_triplet in pos_vocab.w2i.keys() else UNK for e in root_first_seq] + [PAD for i in range(150 - len(root_first_seq))])

            T = Ts[i][:max_src_len, :max_src_len]
            L = Ls[i][:max_src_len, :max_src_len]

            adj = torch.logical_or(L.eq(1), L.eq(0))
            adj = torch.logical_or(L.eq(-1), adj)
            ast_seq = self.ast_data[i][0]
            nl = self.nl_data[i]
            num_node = min(len(root_first_seq), max_src_len)

            ast_vec = self.convert_ast_to_tensor(ast_seq)
            nl_vec = self.convert_nl_to_tensor(nl)

            width = 8
            height = 16
            tree_pos = self.gen_tree_positions(root_first_seq, width, height)
            tree_pos_lst = []
            for k, v in tree_pos.items():
                if len(v) > width * height:
                    v = v[len(v) - width * height :]
                    assert len(v) == width * height
                v = torch.cat([torch.zeros(width * height - len(v)), v])
                tree_pos_lst.append(v)

            tree_pos = torch.stack(tree_pos_lst)

            data = Data(src_seq=ast_vec, L=L, T=T, adj=adj, tree_pos=tree_pos, num_node=num_node, tgt_seq=nl_vec[:-1], target=nl_vec[1:], triplet=triplet)
            edges_data.append(data)
        split_path = f"{self.out_data_path}.pt"
        torch.save(edges_data, split_path)

    def __getitem__(self, index):
        return self.edges_data[index], self.edges_data[index].target
