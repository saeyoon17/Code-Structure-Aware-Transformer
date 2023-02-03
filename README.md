# Code-Structure-Aware-Transformer
This is a replication package for CSA-Trans.
Through the repository, you are able to run all experiments in "CSA-Trans: Code Structure Aware Positional Encoding for AST". To replicate the results, follow the following steps.

## 1. Prepare dataset
- If you want to build the dataset for yourself, first download Python and Java dataset from [dataset link](https://github.com/wasiahmad/NeuralCodeSum/tree/master/data) and put them inside /py and /java directories. Also, download each tree-sitter parser for [python](https://github.com/tree-sitter/tree-sitter-python) and [java](https://github.com/tree-sitter/tree-sitter-java) under directory named tree_sitter. The tree_sitter directory should be outside CSA-Trans directory. tree_sitter_parse.ipynb in each /py and /java guides through AST parsing for each languages, generating tree_sitter_python and tree_sitter_java directories.

- We provide the parsed ASTs in anonymous [link](https://figshare.com/s/f99c4cfda5f78bd04406).

## 2. Preprocess.
For preprocessing Java / Python dataset, set work_dir in process.py as either 'tree_sitter_java' or 'tree_sitter_python'. Run 
- python process.py -data_dir ./ -max_ast_len 150 -process -make_vocab

## 3. Running experiments.
### For single GPU, run 
- python main.py --config=./config/python.py --exp_type summary --g 0
### For multi GPU, (4 GPUs are used for experiments) run
- python -u -m torch.distributed.launch --nproc_per_node 4 --use_env main.py --config=./config/python.py --exp_type summary --g 0,1,2,3

## 4. Comparing with AST-Trans and CodeScribe

### For comparison with ast-trans for python dataset
1. Uncomment ignore_idx in process.py 
2. Set processed_path to ./processed_ast_trans_data/. 
3. Run process.py 
4. Run python_compare_asttrans.py.

### For comparison with CodeScribe
1. Copy each ast.original in train/test/dev to compare_codescribe_{language} train/test/dev.
2. Run process.py with languages = ["compare_codescribe_java/"] / ["compare_codescribe_python/"].
3. Run python_compare_codescribe.py or java_compare_codescribe.py.
