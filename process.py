import argparse
import json
import os

from tqdm import tqdm
from my_ast import MyAst
from utils.vocab import create_vocab

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", default="./", type=str)
parser.add_argument("-max_ast_len", default=250, type=int)
parser.add_argument("-process", action="store_true")
parser.add_argument("-make_vocab", action="store_true")


def skip_code_and_nl_with_skip_id(data_dir, output_dir, ignore_idx):
    # skip data.
    nls = []
    with open(data_dir + "nl.original", "r") as f:
        for line_index, line in enumerate(f.readlines()):
            if line_index in ignore_idx:
                continue
            nls.append(line)

    with open(output_dir + "nl.original", "w") as f:
        for index, nl in tqdm(enumerate(nls), desc="skip nl"):
            nl = "".join(nl)
            f.write(nl)


def process(data_dir, max_len, output_path, lang):

    # For comparison with ast-trans
    # if "/test/" in data_dir:
    #     ignore_idx = [10906, 18378]
    # elif "/dev/" in data_dir:
    #     ignore_idx = [5193, 17919]
    # else:
    #     ignore_idx = [27782, 33412, 40401, 41539, 49342, 53389]
    ignore_idx = []

    with open(data_dir + "ast.original", "r", errors="replace") as f:
        asts = []
        for idx, line in enumerate(f.readlines()):
            if idx in ignore_idx:
                continue
            ast_json = json.loads(line)
            asts.append(ast_json)

    prev_asts_num = len(asts)
    asts = [ast for i, ast in enumerate(asts)]
    next_asts_num = len(asts)
    print(f"filtered {prev_asts_num - next_asts_num} asts")

    root_list = MyAst.process_ast(asts, max_size=max_len)
    MyAst.collect_matrices_and_save(
        root_list,
        output_path + "split_matrices.npz",
        output_path + "split_pot.seq",
        output_path + "idents.seq",
        lang,
        max_len,
    )
    skip_code_and_nl_with_skip_id(data_dir, output_path, ignore_idx)


if __name__ == "__main__":
    args = parser.parse_args()
    data_set_dir = args.data_dir
    max_ast_len = args.max_ast_len

    languages = ["tree_sitter_java/"]
    data_sets = ["dev/", "test/", "train/"]

    if args.process:
        for lang in languages:
            for data_set in data_sets:
                data_path = data_set_dir + lang + data_set
                print("*" * 5, "Process ", data_path, "*" * 5)
                processed_path = data_set_dir + "processed/" + lang + data_set
                os.makedirs(processed_path, exist_ok=True)
                process(data_path, max_ast_len, processed_path, lang)

    if args.make_vocab:
        for lang in languages:
            create_vocab(data_dir=data_set_dir + "processed/" + lang)
