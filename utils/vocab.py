import json
import logging
import os
import pickle
import unicodedata
from collections import Counter

from tqdm import tqdm

PAD = 0
UNK = 1
BOS = 2
EOS = 3

SELF_WORD = "<self>"
PAD_WORD = "<pad>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

log = logging.getLogger()

__all__ = [
    "Vocab",
    "PAD",
    "BOS",
    "EOS",
    "UNK",
    "EOS_WORD",
    "BOS_WORD",
    "PAD_WORD",
    "SELF_WORD",
    "create_vocab",
    "load_vocab",
]


class Vocab(object):
    def __init__(self, need_bos, file_path):
        if not need_bos:
            self.w2i = {PAD_WORD: PAD, UNK_WORD: UNK}
            self.i2w = {PAD: PAD_WORD, UNK: UNK_WORD}
        else:
            self.w2i = {PAD_WORD: PAD, UNK_WORD: UNK, BOS_WORD: BOS, EOS_WORD: EOS}
            self.i2w = {PAD: PAD_WORD, UNK: UNK_WORD, BOS: BOS_WORD, EOS: EOS_WORD}
        self.file_path = file_path

    @staticmethod
    def normalize(token):
        return unicodedata.normalize("NFD", token)

    def size(self):
        return len(self.w2i)

    def add(self, token, is_edge_vocab=False):
        if not is_edge_vocab:
            token = self.normalize(token)
        if token not in self.w2i:
            index = len(self.w2i)
            self.w2i[token] = index
            self.i2w[index] = token

    def add_tokens(self, tokens, is_edge_vocab):
        for token in tokens:
            self.add(token, is_edge_vocab)

    def generate_dict(self, tokens, max_vocab_size=-1, is_edge_vocab=False):
        if is_edge_vocab:
            word_counter = Counter([x for x in tokens])
        else:
            word_counter = Counter([x for c in tokens for x in c])
        if max_vocab_size < 0:
            words = [x[0] for x in word_counter.most_common()]
        else:
            words = [x[0] for x in word_counter.most_common(max_vocab_size - len(self.w2i))]
        self.add_tokens(words, is_edge_vocab)

        self.save()

    def save(self):
        pickle.dump(self.w2i, open(self.file_path, "wb"))

    def load(self):
        self.w2i = pickle.load(open(self.file_path, "rb"))
        self.i2w = {v: k for k, v in self.w2i.items()}


class CTVocab(object):
    def __init__(self, need_bos, ct_path, file_path):
        if not need_bos:
            self.w2i = {PAD_WORD: PAD, UNK_WORD: UNK}
            self.i2w = {PAD: PAD_WORD, UNK: UNK_WORD}
        else:
            self.w2i = {PAD_WORD: PAD, UNK_WORD: UNK, BOS_WORD: BOS, EOS_WORD: EOS}
            self.i2w = {PAD: PAD_WORD, UNK: UNK_WORD, BOS: BOS_WORD, EOS: EOS_WORD}
        self.ct_path = ct_path
        self.file_path = file_path

    @staticmethod
    def normalize(token):
        return unicodedata.normalize("NFD", token)

    def size(self):
        return len(self.w2i)

    def add(self, token):
        if token not in self.w2i:
            index = len(self.w2i)
            self.w2i[token] = index
            self.i2w[index] = token

    def add_tokens(self, tokens):
        for token in tokens:
            self.add(token)

    def generate_dict(self):
        with open(self.ct_path, "r") as f:
            vocab_dict = json.load(f)
        for key in vocab_dict.keys():
            self.add(key)
        self.save()

    def save(self):
        pickle.dump(self.w2i, open(self.file_path, "wb"))

    def load(self):
        self.w2i = pickle.load(open(self.file_path, "rb"))
        self.i2w = {v: k for k, v in self.w2i.items()}


def load_vocab(data_dir, data_type, ctvocab=False):
    log.info(f"load vocab from {data_dir}")
    split_str = "split_ast_vocab.pkl"
    # TODO find out what these are for
    if data_type in ["pot"]:
        src_vocab = Vocab(need_bos=False, file_path=data_dir + "/vocab/" + split_str)
        src_vocab.load()

    if not ctvocab:
        nl_vocab = Vocab(need_bos=True, file_path=data_dir + "/vocab/" + "nl_vocab.pkl")
        nl_vocab.load()
    else:
        nl_vocab = CTVocab(
            need_bos=True,
            ct_path=data_dir + "/vocab/" + "ct_vocab.json",
            file_path=data_dir + "/vocab/" + "nl_vocab.pkl",
        )
        nl_vocab.generate_dict()
        nl_vocab.load()

    return (src_vocab, nl_vocab)


def create_vocab(data_dir):
    # create vocab
    log.info("init vocab")
    output_dir = data_dir + "vocab/"
    os.makedirs(output_dir, exist_ok=True)

    split_ast_tokens = []
    with open(data_dir + "train/" + "split_pot.seq", "r") as f:
        for line in tqdm(f.readlines(), desc="loading ast from train ...."):
            line = eval(line)
            line = line[0]
            split_ast_tokens.append([e.split(":")[1] for e in line])
    with open(data_dir + "dev/" + "split_pot.seq", "r") as f:
        for line in tqdm(f.readlines(), desc="loading ast from dev ...."):
            line = eval(line)
            line = line[0]
            split_ast_tokens.append([e.split(":")[1] for e in line])
    # split_ast_tokens.append(["self", "type2type", "type2idt", "idt2type"])
    split_ast_vocab = Vocab(need_bos=False, file_path=output_dir + "split_ast_vocab.pkl")
    split_ast_vocab.generate_dict(split_ast_tokens, 10000)

    # ident_tokens = []
    # with open(data_dir + "train/" + "idents.seq", "r") as f:
    #     for line in tqdm(f.readlines(), desc="loading ast from train ...."):
    #         ident_tokens.append([e for e in eval(line)])
    # with open(data_dir + "dev/" + "idents.seq", "r") as f:
    #     for line in tqdm(f.readlines(), desc="loading ast from dev ...."):
    #         ident_tokens.append([e for e in eval(line)])
    # ident_vocab = Vocab(need_bos=False, file_path=output_dir + "ident_vocab.pkl")
    # ident_vocab.generate_dict(ident_tokens)

    nl_tokens = []
    with open(data_dir + "train/nl.original", "r") as f:
        for line in tqdm(f.readlines(), desc="loading nl from train ...."):
            nl_tokens.append(line.split())
    with open(data_dir + "dev/nl.original", "r") as f:
        for line in tqdm(f.readlines(), desc="loading nl from dev ...."):
            nl_tokens.append(line.split())
    nl_vocab = Vocab(need_bos=True, file_path=output_dir + "nl_vocab.pkl")
    nl_vocab.generate_dict(nl_tokens, 20000)

    # def update_node_child_idx(root_node):
    #     for idx, e in enumerate(root_node.children):
    #         if e.label.split(":")[0] == "idx":
    #             e.child_idx = -1
    #         else:
    #             e.child_idx = idx
    #         update_node_child_idx(e)

    # def get_node_triplet(root_node):
    #     for idx, e in enumerate(root_node.children):
    #         e.node_triplet = str((e.level, e.parent.child_idx, e.child_idx))
    #         get_node_triplet(e)

    # train_processed_path = "../cbgt/processed/tree_sitter_java/train/split_matrices.npz"
    # dev_processed_path = "../cbgt/processed/tree_sitter_java/dev/split_matrices.npz"

    # import numpy as np

    # train_data = np.load(train_processed_path, allow_pickle=True)
    # train_rfs = train_data["root_first_seq"]

    # dev_data = np.load(dev_processed_path, allow_pickle=True)
    # dev_rfs = dev_data["root_first_seq"]
    # rfs = list(train_rfs) + list(dev_rfs)

    # triplets = []
    # for ast in tqdm(rfs):
    #     rftriplet = []
    #     ast[0].child_idx = 0
    #     ast[0].node_triplet = str((0, 0, 0))
    #     update_node_child_idx(ast[0])
    #     get_node_triplet(ast[0])
    #     rftriplet = [e.node_triplet for e in ast]
    #     triplets.append(rftriplet)
    # pos_vocab = Vocab(need_bos=False, file_path="node_triplet_dictionary_java.pt")
    # pos_vocab.generate_dict(triplets)

    print(f"split ast vocab size: {len(split_ast_vocab.w2i)} \n")
    # print(f"ident vocab size: {len(ident_vocab.w2i)} \n")
    print(f"nl vocab size: {len(nl_vocab.w2i)} \n")
    # print(f"pos vocab size: {pos_vocab.size()}")
