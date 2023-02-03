import copy
import re
import torch
import joblib
import networkx as nx
import numpy as np
from tqdm import tqdm

__all__ = ["split_variable", "PathExtract", "MyAst"]

PAD = 0
UNK = 1
BOS = 2
EOS = 3

SELF_WORD = "<self>"
PAD_WORD = "<pad>"
UNK_WORD = "<unk>"
BOS_WORD = "<s>"
EOS_WORD = "</s>"

n_jobs = 30


class Node:
    def __init__(
        self,
        label="",
        parent=None,
        is_simple_name=False,
        children=[],
        child_idx=-1,
        start_lineno=-1,
        end_lineno=-1,
    ):
        self.label = label
        self.parent = parent
        self.children = children
        self.is_simple_name = is_simple_name
        self.child_idx = child_idx
        self.level = 0
        self.start_lineno = start_lineno
        self.end_lineno = end_lineno


class MyAst:
    @staticmethod
    def process_ast(asts, max_size=-1):
        parallel = joblib.Parallel(n_jobs=n_jobs)
        func = joblib.delayed(MyAst.__process_treesitter)

        root_nodes = parallel(func(ast, max_size) for ast in tqdm(asts, desc=f"process AST: size {max_size}"))
        return root_nodes

    @staticmethod
    def collect_seq_and_save(root_nodes, output_file, seq_type):
        parallel = joblib.Parallel(n_jobs=n_jobs)
        if seq_type == "sbt":
            func = joblib.delayed(MyAst.__get_sbt_seq)
        elif seq_type == "pot":
            func = joblib.delayed(MyAst.__get_pot_seq)
        else:
            raise Exception("Invalid seq_type, must be in [sbt, pot]")

        seqs = parallel(func(root_node) for root_node in tqdm(root_nodes, desc="generate " + seq_type))

        with open(output_file, "w") as f:
            for line_index, line in enumerate(seqs):
                f.write(str(line) + ("" if line_index == len(seqs) - 1 else "\n"))

    @staticmethod
    def collect_matrices_and_save(root_nodes, output_matrices_file, output_pot_file, output_path, lang, max_size):
        parallel = joblib.Parallel(n_jobs=n_jobs)
        func = joblib.delayed(MyAst.__get_matrices)

        matrices = parallel(func(root_node, lang, max_size) for root_node in tqdm(root_nodes, desc="generate matrices"))

        (
            root_first_seq,
            L,
            T,
            root_first_level,
            pot_seq,
            par_edges,
            bro_edges,
        ) = list(zip(*matrices))

        np.savez(
            output_matrices_file,
            root_first_seq=root_first_seq,
            root_first_level=root_first_level,
            L=L,
            T=T,
            parent=list(par_edges),
            brother=list(bro_edges),
        )

        with open(output_pot_file, "w") as f:
            for line_index, line in enumerate(pot_seq):
                f.write(str(line) + ("" if line_index == len(pot_seq) - 1 else "\n"))

    @staticmethod
    def __process_treesitter(ast_json, max_size=-1):

        node_num = len(ast_json)
        node_list = [copy.deepcopy(Node()) for i in range(node_num)]

        for i in range(node_num):
            node_attr = ast_json[i]
            node = node_list[i]
            tmp = node_attr["label"].split(":")
            node.label = ":".join(tmp[:-3] + [tmp[-1]])
            node.start_lineno = int(tmp[-3])
            node.end_lineno = int(tmp[-2])

            if "children" in node_attr:
                for child_idx, child_id in enumerate(node_attr["children"]):
                    # idx starts from 1 :(
                    child_id = int(child_id.split(":")[-1]) - 1
                    node_list[child_id].parent = node
                    node.children.append(node_list[child_id])
                    node_list[child_id].child_idx = child_idx

        if max_size > 0:
            MyAst.__sub_tree(node_list[0], max_size)
        return node_list[0]

    @staticmethod
    def __sub_tree(root_node, max_size, i=0):
        root_node.num = i
        i = i + 1
        if i > max_size:
            return -1
        else:
            for j, child in enumerate(root_node.children):
                i = MyAst.__sub_tree(child, max_size, i)
                if i == -1:
                    root_node.children = root_node.children[:j]
                    return -2
                if i == -2:
                    root_node.children = root_node.children[: j + 1]
                    return i
            return i

    @staticmethod
    def __get_root_first_seq(root_node):
        li = [root_node]
        for child in root_node.children:
            li += MyAst.__get_root_first_seq(child)
        return li

    @staticmethod
    def __get_pot_seq(root_node):
        root_first_seq = MyAst.__get_root_first_seq(root_node)
        root_first_labels = [":".join(node.label.split(":")[1:-1]) for node in root_first_seq]
        return root_first_labels

    @staticmethod
    def build_networkx_graph(root_first_seq):
        G = nx.Graph()
        for node in root_first_seq:
            G.add_node(node.label + ":" + str(node.num))
        root_first_seq[0].level = 0
        # parent always come first in pre order traversal
        for node in root_first_seq:
            if node.parent != None:
                G.add_edge(
                    node.parent.label + ":" + str(node.parent.num),
                    node.label + ":" + str(node.num),
                )
                node.level = node.parent.level + 1
        return G

    @staticmethod
    def __get_labels(root_first_seq):
        root_first_node_labels = []
        for idx, e in enumerate(root_first_seq):
            tmp = e.label.split(":")
            node = tmp[0]
            val = tmp[1]
            if node == "type":
                root_first_node_labels.append(e.label)
            else:
                root_first_node_labels.append(e.label)
        return (root_first_node_labels,)

    @staticmethod
    def __get_distance_pairs(path):
        node_num = len(path)
        distance_pairs = {}
        if node_num >= 2:
            for i in range(node_num - 1):
                for j in range(i + 1, node_num):
                    distance_pairs[(path[i], path[j])] = j - i
        return distance_pairs

    @staticmethod
    def __get_matrices(root_node, lang, max_size):
        root_first_seq = MyAst.__get_root_first_seq(root_node)

        def get_node_levels(rfs):
            levels = []
            for e in rfs:
                level = 0
                tmp = e
                while tmp.parent != None:
                    level += 1
                    tmp = tmp.parent

                levels.append(level)
            return levels

        def get_node_types(rfs):
            node_types = []
            for e in rfs:
                if e.label.split(":")[0] == "nont":
                    node_types.append(0)
                else:
                    node_types.append(1)
            return node_types

        rfl = MyAst.__get_labels(root_first_seq)
        MyAst.__sub_tree(root_first_seq[0], max_size)
        G = MyAst.build_networkx_graph(root_first_seq)
        root_first_level = get_node_levels(root_first_seq)
        root_first_level = root_first_level + [0 for i in range(max_size - len(root_first_seq))]

        distance_map = {}
        brother_map = {}

        parent_path_list = []
        brother_path_list = []

        for node in root_first_seq:
            if len(node.children) == 0:
                path = [node.num]
                n = node
                while n.parent is not None:
                    path.append(n.parent.num)
                    n = n.parent
                    parent_path_list.append(list(reversed(path)))
            else:
                brother_path_list.append([child.num for child in node.children])

        for path in parent_path_list:
            distance_map.update(MyAst.__get_distance_pairs(path))

        for path in brother_path_list:
            brother_map.update(MyAst.__get_distance_pairs(path))

        # parent
        L = torch.zeros(max_size, max_size)
        # sibling
        T = torch.zeros(max_size, max_size)
        for pair, length in distance_map.items():
            if pair[0] < max_size and pair[1] < max_size:
                L[pair[0], pair[1]] = length
                L[pair[1], pair[0]] = -length

        for pair, length in brother_map.items():
            if pair[0] < max_size and pair[1] < max_size:
                T[pair[0], pair[1]] = length
                T[pair[1], pair[0]] = -length

        return (
            root_first_seq,
            L,
            T,
            root_first_level,
            rfl,
            distance_map,
            brother_map,
        )


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def split_variable(name):
    def camel_case_split(identifier):
        matches = re.finditer(
            ".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)",
            identifier,
        )
        return [m.group(0) for m in matches]

    blocks = []
    for underscore_block in name.split("_"):
        blocks.extend(camel_case_split(underscore_block))

    return [block.lower() for block in blocks]
