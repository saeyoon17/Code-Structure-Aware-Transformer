#!/usr/bin/env python
# coding: utf-8

# # Experiment for RQ2: Intermediate node prediction

# ### We first build the dataset for intermediate node prediction

# In[1]:


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# In[2]:


import torch
import numpy as np
from itertools import combinations
from utils import load_vocab
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
from py_config_runner import ConfigObject
from module import CSATrans
from ignite.utils import convert_tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F


# ## random 3-hop nodes

# In[3]:


matrices_path = "./processed/tree_sitter_java/test/split_matrices.npz"
data = np.load(matrices_path, allow_pickle=True)
test_rfs = data["root_first_seq"]
import networkx as nx

from collections import defaultdict

result = defaultdict(lambda: defaultdict(lambda: []))
import time

start_time = time.time()
for num_hop in [3, 5, 7]:
    for i in range(1):
        print(f"===== Start for {num_hop} - {i} - time: {(time.time() - start_time)/60}===== ")

        def generate_graph(node, G):
            G.add_node(node.label)
            for e in node.children:
                # do not add identifier nodes
                generate_graph(e, G)
                G.add_edge(node.label, e.label)

        for split_idx, rfs in enumerate([test_rfs]):
            original_test_dataset = {}
            for data_idx, sample_ast in enumerate(tqdm(rfs)):
                G = nx.Graph()
                generate_graph(sample_ast[0], G)
                paths = dict(nx.all_pairs_shortest_path(G, cutoff=num_hop))
                cands = []
                for start_node in paths.keys():
                    for end_node in paths[start_node].keys():
                        path = paths[start_node][end_node]
                        if len(path) == num_hop and (int(start_node.split(":")[-1]) < int(end_node.split(":")[-1])):
                            cands.append(path)

                SAMPLE_NUM = min([10, len(cands)])
                if SAMPLE_NUM > 0:
                    idx1 = np.random.choice(range(len(cands)), size=SAMPLE_NUM, replace=False)
                    sample_cands = [cands[i] for i in idx1]
                    original_test_dataset[data_idx] = sample_cands
                else:
                    original_test_dataset[data_idx] = []
        # get dataset stat
        whole = list(original_test_dataset.values())
        acc = 0
        for e in whole:
            acc += len(e)
        print(f"num_hop: {num_hop}, total data num: {acc}")
        continue

        # ## retrieve parent-child relationships

        # In[4]:

        def _graph_prepare_batch(batch, device=None, non_blocking: bool = False):
            x, y = batch
            return (
                x.to(device),
                convert_tensor(y, device=device, non_blocking=non_blocking),
            )

        class SyntheticDataset(Dataset):
            def __init__(self, embeddings, targets):
                self.x = embeddings
                self.y = targets

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                x = self.x[idx]
                y = self.y[idx]
                return x, y

        class MLP(nn.Module):
            def __init__(self, indim, hidden, outdim):
                super().__init__()
                self.fc1 = nn.Linear(indim, hidden)
                self.fc2 = nn.Linear(hidden, hidden)
                self.fc3 = nn.Linear(hidden, hidden)
                self.fc4 = nn.Linear(hidden, outdim)
                self.dropout = nn.Dropout(0.2)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(F.relu(self.fc2(x)))
                x = self.dropout(F.relu(self.fc3(x)))
                x = F.relu(self.fc4(x))
                return x

        # ## Node prediction for CSA-Trans

        # In[5]:

        parser = argparse.ArgumentParser("Example application")
        parser.add_argument("--config", type=Path, help="Input configuration file")
        parser.add_argument("--use_hype_params", action="store_true")
        parser.add_argument("--data_type", type=str, default="")
        parser.add_argument("--exp_type", type=str, default="summary")
        parser.add_argument("--g", type=str, default="")

        args = parser.parse_args(
            [
                "--config",
                "./config/java.py",
                "--g",
                "0",
            ]
        )
        config = ConfigObject(args.config)

        if args.g != "":
            config.device = "cuda"
            config.g = args.g

        config.data_dir = "./processed/tree_sitter_java"
        (
            config.src_vocab,
            config.tgt_vocab,
        ) = load_vocab(config.data_dir, config.is_split, config.data_type)
        test_data_set = config.data_set(config, "test")
        test_loader = DataLoader(
            dataset=test_data_set,
            batch_size=config.batch_size // len(config.g.split(",")),
            shuffle=False,
            collate_fn=test_data_set.collect_fn,
        )

        # In[6]:

        model = config.model(
            config.src_vocab.size(),
            config.tgt_vocab.size(),
            config.hidden_size,
            config.num_heads,
            config.num_layers,
            config.sbm_layers,
            config.use_pegen,
            config.dim_feed_forward,
            config.dropout,
            config.pe_dim,
            config.pegen_dim,
            config.sbm_enc_dim,
            config.clusters,
            config.full_att,
            config.checkpoint,
            "java",
            config.max_src_len,
        )
        state_path = "./outputs/java.pt"
        state_dict = torch.load(state_path)
        model.load_state_dict(state_dict)
        model = model.to("cuda")

        # In[7]:

        src_pes = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_loader)):
                x, y = _graph_prepare_batch(batch)
                y_, sparsity, src_pe, graphs, attns = model(x.to("cuda"))
                src_pe = src_pe.detach()
                src_pes += src_pe

        # In[8]:

        BATCH_SIZE = 128
        input_dim = 256
        hidden_dim = 1024
        device = "cuda"
        criterion = nn.CrossEntropyLoss()

        # In[9]:

        whole = list(original_test_dataset.values())
        X = []  # the two embeddings
        Y = []  # intermediate vocabulary
        num_to_predict = num_hop - 2
        for ast_idx, ast_instance in tqdm(enumerate(whole)):
            for path in ast_instance:
                try:
                    nodes = []

                    node1 = src_pes[ast_idx][int(path[0].split(":")[-1]) - 1]
                    node2 = src_pes[ast_idx][int(path[-1].split(":")[-1]) - 1]

                    if path[1].split(":")[1] not in config.src_vocab.w2i.keys():
                        continue

                    tgts = []
                    for i in range(num_to_predict):
                        tgts.append(config.src_vocab.w2i[path[i + 1].split(":")[1]])

                    X.append(torch.cat([node1, node2]))
                    Y.append(torch.tensor([tgts]))
                except:
                    # OOV error
                    continue
        train_X = X[: int(len(X) * 0.8)]
        train_Y = Y[: int(len(X) * 0.8)]
        test_X = X[int(len(X) * 0.8) :]
        test_Y = Y[int(len(X) * 0.8) :]
        train_dataset = SyntheticDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = SyntheticDataset(test_X, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # In[10]:

        # In[11]:

        train_X[0].size()

        # In[12]:

        out_dim = config.src_vocab.size() * num_to_predict
        model = MLP(input_dim, hidden_dim, out_dim)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"num_param: {num_param}")

        model.train()
        for epoch in range(30):
            loss_acc = 0
            for batch in train_loader:
                x, y = batch
                pred = model(x.to(device))
                bsize = x.size(0)
                pred = pred.reshape(bsize, 10000, -1)
                loss = criterion(pred, y.to(device).squeeze(-2))
                loss.backward()
                loss_acc += loss.item()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 5 == 0:
                per_sample = loss_acc / len(train_loader) / BATCH_SIZE
                total_correct = 0
                model.eval()
                for batch in test_loader:
                    x, y = batch
                    pred = model(x.to(device))
                    bsize = x.size(0)
                    total_correct += torch.sum(
                        torch.all(
                            (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                            dim=-1,
                        )
                    )
                print(f"Epoch - {epoch}, Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
                model.train()

        # In[13]:

        total_correct = 0
        model.eval()
        for batch in tqdm(test_loader):
            x, y = batch
            pred = model(x.to(device))
            bsize = x.size(0)
            total_correct += torch.sum(
                torch.all(
                    (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                    dim=-1,
                )
            )
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
        result[num_hop]["csa-trans"].append(total_correct / len(test_loader) / BATCH_SIZE)

        # ## Treepos

        # In[14]:

        args = parser.parse_args(
            [
                "--config",
                "./config/java_treepos.py",
                "--g",
                "0",
            ]
        )
        config = ConfigObject(args.config)
        config.data_dir = "./processed/tree_sitter_java"
        (
            config.src_vocab,
            config.tgt_vocab,
        ) = load_vocab(config.data_dir, config.is_split, config.data_type)
        model = config.model(
            config.src_vocab.size(),
            config.tgt_vocab.size(),
            config.hidden_size,
            config.num_heads,
            config.num_layers,
            config.sbm_layers,
            config.use_pegen,
            config.dim_feed_forward,
            config.dropout,
            config.pe_dim,
            config.pegen_dim,
            config.sbm_enc_dim,
            config.clusters,
            config.full_att,
            config.checkpoint,
            "java",
            config.max_src_len,
        )
        test_data_set = config.data_set(config, "test")
        test_loader = DataLoader(
            dataset=test_data_set,
            batch_size=config.batch_size // len(config.g.split(",")),
            shuffle=False,
            collate_fn=test_data_set.collect_fn,
        )
        state_path = "./outputs/java_treepos.pt"
        state_dict = torch.load(state_path)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to("cuda")

        # In[15]:

        src_pes = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_loader)):
                x, y = _graph_prepare_batch(batch)
                y_, sparsity, src_pe, _, _ = model(x.to("cuda"))
                src_pe = src_pe.detach()
                src_pes += src_pe

        # In[16]:

        whole = list(original_test_dataset.values())
        X = []  # the two embeddings
        Y = []  # intermediate vocabulary
        for ast_idx, ast_instance in tqdm(enumerate(whole)):
            for path in ast_instance:
                try:
                    nodes = []

                    node1 = src_pes[ast_idx][int(path[0].split(":")[-1]) - 1]
                    node2 = src_pes[ast_idx][int(path[-1].split(":")[-1]) - 1]

                    if path[1].split(":")[1] not in config.src_vocab.w2i.keys():
                        continue

                    tgts = []
                    for i in range(num_to_predict):
                        tgts.append(config.src_vocab.w2i[path[i + 1].split(":")[1]])

                    X.append(torch.cat([node1, node2]))
                    Y.append(torch.tensor([tgts]))
                except:
                    # OOV error
                    continue
        train_X = X[: int(len(X) * 0.8)]
        train_Y = Y[: int(len(X) * 0.8)]
        test_X = X[int(len(X) * 0.8) :]
        test_Y = Y[int(len(X) * 0.8) :]
        train_dataset = SyntheticDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = SyntheticDataset(test_X, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # In[17]:

        out_dim = config.src_vocab.size() * num_to_predict
        model = MLP(input_dim, hidden_dim, out_dim)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"num_param: {num_param}")

        model.train()
        for epoch in range(30):
            loss_acc = 0
            for batch in train_loader:
                x, y = batch
                pred = model(x.to(device))
                bsize = x.size(0)
                pred = pred.reshape(bsize, 10000, -1)
                loss = criterion(pred, y.to(device).squeeze(-2))
                loss.backward()
                loss_acc += loss.item()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 5 == 0:
                per_sample = loss_acc / len(train_loader) / BATCH_SIZE
                total_correct = 0
                model.eval()
                for batch in test_loader:
                    x, y = batch
                    pred = model(x.to(device))
                    bsize = x.size(0)
                    total_correct += torch.sum(
                        torch.all(
                            (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                            dim=-1,
                        )
                    )
                print(f"Epoch - {epoch}, Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
                model.train()

        # In[18]:

        total_correct = 0
        model.eval()
        for batch in tqdm(test_loader):
            x, y = batch
            pred = model(x.to(device))
            bsize = x.size(0)
            total_correct += torch.sum(
                torch.all(
                    (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                    dim=-1,
                )
            )
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
        result[num_hop]["treepos"].append(total_correct / len(test_loader) / BATCH_SIZE)

        # ## Laplacian

        # ### Since laplacian pe and sequential pe are not learnable, they can be just used by changing the configs from treepos configuration

        # In[19]:

        args = parser.parse_args(
            [
                "--config",
                "./config/java_lap.py",
                "--g",
                "0",
            ]
        )
        config = ConfigObject(args.config)
        config.data_dir = "./processed/tree_sitter_java"
        (
            config.src_vocab,
            config.tgt_vocab,
        ) = load_vocab(config.data_dir, config.is_split, config.data_type)
        model = config.model(
            config.src_vocab.size(),
            config.tgt_vocab.size(),
            config.hidden_size,
            config.num_heads,
            config.num_layers,
            config.sbm_layers,
            config.use_pegen,
            config.dim_feed_forward,
            config.dropout,
            config.pe_dim,
            config.pegen_dim,
            config.sbm_enc_dim,
            config.clusters,
            config.full_att,
            config.checkpoint,
            "java",
            config.max_src_len,
        )
        test_data_set = config.data_set(config, "test")
        test_loader = DataLoader(
            dataset=test_data_set,
            batch_size=config.batch_size // len(config.g.split(",")),
            shuffle=False,
            collate_fn=test_data_set.collect_fn,
        )

        # In[20]:

        state_path = "./outputs/java_lap.pt"
        state_dict = torch.load(state_path)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to("cuda")

        # In[21]:

        src_pes = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_loader)):
                x, y = _graph_prepare_batch(batch)
                y_, sparsity, src_pe, _, _ = model(x.to("cuda"))
                src_pe = src_pe.detach()
                src_pes += src_pe

        # In[22]:

        whole = list(original_test_dataset.values())
        X = []  # the two embeddings
        Y = []  # intermediate vocabulary
        for ast_idx, ast_instance in tqdm(enumerate(whole)):
            for path in ast_instance:
                try:
                    nodes = []

                    node1 = src_pes[ast_idx][int(path[0].split(":")[-1]) - 1]
                    node2 = src_pes[ast_idx][int(path[-1].split(":")[-1]) - 1]

                    if path[1].split(":")[1] not in config.src_vocab.w2i.keys():
                        continue

                    tgts = []
                    for i in range(num_to_predict):
                        tgts.append(config.src_vocab.w2i[path[i + 1].split(":")[1]])

                    X.append(torch.cat([node1, node2]))
                    Y.append(torch.tensor([tgts]))
                except:
                    # OOV error
                    continue
        train_X = X[: int(len(X) * 0.8)]
        train_Y = Y[: int(len(X) * 0.8)]
        test_X = X[int(len(X) * 0.8) :]
        test_Y = Y[int(len(X) * 0.8) :]
        train_dataset = SyntheticDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = SyntheticDataset(test_X, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # In[23]:

        out_dim = config.src_vocab.size() * num_to_predict
        model = MLP(input_dim, hidden_dim, out_dim)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-4)
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"num_param: {num_param}")

        model.train()
        for epoch in range(30):
            loss_acc = 0
            for batch in train_loader:
                x, y = batch
                pred = model(x.to(device))
                bsize = x.size(0)
                pred = pred.reshape(bsize, 10000, -1)
                loss = criterion(pred, y.to(device).squeeze(-2))
                loss.backward()
                loss_acc += loss.item()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 5 == 0:
                per_sample = loss_acc / len(train_loader) / BATCH_SIZE
                total_correct = 0
                model.eval()
                for batch in test_loader:
                    x, y = batch
                    pred = model(x.to(device))
                    bsize = x.size(0)
                    # import ipdb
                    # ipdb.set_trace()
                    total_correct += torch.sum(
                        torch.all(
                            (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                            dim=-1,
                        )
                    )
                print(f"Epoch - {epoch}, Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
                model.train()

        # In[24]:

        total_correct = 0
        model.eval()
        for batch in tqdm(test_loader):
            x, y = batch
            pred = model(x.to(device))
            bsize = x.size(0)
            total_correct += torch.sum(
                torch.all(
                    (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                    dim=-1,
                )
            )
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
        result[num_hop]["laplacian"].append(total_correct / len(test_loader) / BATCH_SIZE)

        # ## Sequential

        # In[25]:

        from module.components import PositionalEncoding

        sbm_enc_dim = 768
        src_pe = PositionalEncoding(sbm_enc_dim, config["max_src_len"])
        src_pe = src_pe.pe.squeeze()

        # In[26]:

        whole = list(original_test_dataset.values())
        X = []  # the two embeddings
        Y = []  # intermediate vocabulary
        for ast_idx, ast_instance in tqdm(enumerate(whole)):
            for path in ast_instance:
                try:
                    nodes = []

                    node1 = src_pe[int(path[0].split(":")[-1]) - 1]
                    node2 = src_pe[int(path[-1].split(":")[-1]) - 1]

                    if path[1].split(":")[1] not in config.src_vocab.w2i.keys():
                        continue

                    tgts = []
                    for i in range(num_to_predict):
                        tgts.append(config.src_vocab.w2i[path[i + 1].split(":")[1]])

                    X.append(torch.cat([node1, node2]))
                    Y.append(torch.tensor([tgts]))
                except:
                    # OOV error
                    continue
        train_X = X[: int(len(X) * 0.8)]
        train_Y = Y[: int(len(X) * 0.8)]
        test_X = X[int(len(X) * 0.8) :]
        test_Y = Y[int(len(X) * 0.8) :]
        train_dataset = SyntheticDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = SyntheticDataset(test_X, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # In[27]:

        input_dim = 768 * 2
        hidden_dim = 1024
        out_dim = config.src_vocab.size() * num_to_predict
        model = MLP(input_dim, hidden_dim, out_dim)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"num_param: {num_param}")

        model.train()
        for epoch in range(30):
            loss_acc = 0
            for batch in train_loader:
                x, y = batch
                pred = model(x.to(device))
                bsize = x.size(0)
                pred = pred.reshape(bsize, 10000, -1)
                loss = criterion(pred, y.to(device).squeeze(-2))
                loss.backward()
                loss_acc += loss.item()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 5 == 0:
                per_sample = loss_acc / len(train_loader) / BATCH_SIZE
                total_correct = 0
                model.eval()
                for batch in test_loader:
                    x, y = batch
                    pred = model(x.to(device))
                    bsize = x.size(0)
                    total_correct += torch.sum(
                        torch.all(
                            (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                            dim=-1,
                        )
                    )
                print(f"Epoch - {epoch}, Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
                model.train()

        # In[28]:

        total_correct = 0
        model.eval()
        for batch in tqdm(test_loader):
            x, y = batch
            pred = model(x.to(device))
            bsize = x.size(0)
            total_correct += torch.sum(
                torch.all(
                    (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                    dim=-1,
                )
            )
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
        result[num_hop]["sequential"].append(total_correct / len(test_loader) / BATCH_SIZE)

        # ## Triplet

        # In[29]:

        args = parser.parse_args(
            [
                "--config",
                "./config/java_triplet.py",
                "--g",
                "0",
            ]
        )
        config = ConfigObject(args.config)
        config.data_dir = "./processed/tree_sitter_java"
        (
            config.src_vocab,
            config.tgt_vocab,
        ) = load_vocab(config.data_dir, config.is_split, config.data_type)
        model = config.model(
            config.src_vocab.size(),
            config.tgt_vocab.size(),
            config.hidden_size,
            config.num_heads,
            config.num_layers,
            config.sbm_layers,
            config.use_pegen,
            config.dim_feed_forward,
            config.dropout,
            config.pe_dim,
            config.pegen_dim,
            config.sbm_enc_dim,
            config.clusters,
            config.full_att,
            config.checkpoint,
            "java",
            config.max_src_len,
        )
        test_data_set = config.data_set(config, "test")
        test_loader = DataLoader(
            dataset=test_data_set,
            batch_size=config.batch_size // len(config.g.split(",")),
            shuffle=False,
            collate_fn=test_data_set.collect_fn,
        )

        # In[30]:

        state_path = "./outputs/java_triplet.pt"
        state_dict = torch.load(state_path)
        model.load_state_dict(state_dict)
        model.eval()
        model = model.to("cuda")

        # In[31]:

        src_pes = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(test_loader)):
                # if idx >= 5:
                #     break
                x, y = _graph_prepare_batch(batch)
                y_, sparsity, src_pe, _, _ = model(x.to("cuda"))
                src_pe = src_pe.detach()
                src_pes += src_pe

        # In[32]:

        whole = list(original_test_dataset.values())
        X = []  # the two embeddings
        Y = []  # intermediate vocabulary
        for ast_idx, ast_instance in tqdm(enumerate(whole)):
            for path in ast_instance:
                try:
                    nodes = []

                    node1 = src_pes[ast_idx][int(path[0].split(":")[-1]) - 1]
                    node2 = src_pes[ast_idx][int(path[-1].split(":")[-1]) - 1]

                    if path[1].split(":")[1] not in config.src_vocab.w2i.keys():
                        continue

                    tgts = []
                    for i in range(num_to_predict):
                        tgts.append(config.src_vocab.w2i[path[i + 1].split(":")[1]])

                    X.append(torch.cat([node1, node2]))
                    Y.append(torch.tensor([tgts]))
                except:
                    # OOV error
                    continue
        train_X = X[: int(len(X) * 0.8)]
        train_Y = Y[: int(len(X) * 0.8)]
        test_X = X[int(len(X) * 0.8) :]
        test_Y = Y[int(len(X) * 0.8) :]
        train_dataset = SyntheticDataset(train_X, train_Y)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataset = SyntheticDataset(test_X, test_Y)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # In[33]:

        input_dim = 256
        hidden_dim = 1024
        out_dim = config.src_vocab.size() * num_to_predict
        model = MLP(input_dim, hidden_dim, out_dim)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"num_param: {num_param}")

        model.train()
        for epoch in range(30):
            loss_acc = 0
            for batch in train_loader:
                x, y = batch
                pred = model(x.to(device))
                bsize = x.size(0)
                pred = pred.reshape(bsize, 10000, -1)
                loss = criterion(pred, y.to(device).squeeze(-2))
                loss.backward()
                loss_acc += loss.item()
                optimizer.step()
                optimizer.zero_grad()

            if epoch % 5 == 0:
                per_sample = loss_acc / len(train_loader) / BATCH_SIZE
                total_correct = 0
                model.eval()
                for batch in test_loader:
                    x, y = batch
                    pred = model(x.to(device))
                    bsize = x.size(0)
                    total_correct += torch.sum(
                        torch.all(
                            (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                            dim=-1,
                        )
                    )
                print(f"Epoch - {epoch}, Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
                model.train()

        # In[34]:

        total_correct = 0
        model.eval()
        for batch in tqdm(test_loader):
            x, y = batch
            pred = model(x.to(device))
            bsize = x.size(0)
            total_correct += torch.sum(
                torch.all(
                    (torch.argmax(pred.reshape(bsize, 10000, -1), dim=-2) == y.squeeze(-2).to("cuda")),
                    dim=-1,
                )
            )
        print(f"Total correct: {total_correct}")
        print(f"Accuracy: {total_correct / len(test_loader) / BATCH_SIZE}")
        result[num_hop]["triplet"].append(total_correct / len(test_loader) / BATCH_SIZE)
    import pickle

    with open(f"result_java_{num_hop}.pkl", "wb") as f:
        pickle.dump(dict(result[num_hop]), f)
print(result)
