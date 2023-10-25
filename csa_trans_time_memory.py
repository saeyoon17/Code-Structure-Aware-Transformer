import numpy as np
from itertools import combinations
from tqdm import tqdm
from utils import load_vocab
from torch.utils.data import DataLoader
import torch

import argparse
import os
from pathlib import Path

from py_config_runner import ConfigObject

parser = argparse.ArgumentParser("Example application")
parser.add_argument("--config", type=Path, help="Input configuration file")
parser.add_argument("--use_hype_params", action="store_true")
parser.add_argument("--data_type", type=str, default="")
parser.add_argument("--exp_type", type=str, default="summary")
parser.add_argument("--g", type=str, default="")

args = parser.parse_args(
    [
        "--config",
        "../csa-trans-test/config/python.py",
        "--g",
        "0",
    ]
)
config = ConfigObject(args.config)

if args.g != "":
    config.device = "cuda"
    config.g = args.g

config.data_dir = "./processed/tree_sitter_python"
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
    "python",
    config.max_src_len,
)

state_path = "../csa-trans-test/outputs/python.pt"
state_dict = torch.load(state_path)
model.load_state_dict(state_dict)
model = model.to("cuda")

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"num_param: {num_param}")
model = model.to('cuda')
model.train()

from ignite.utils import convert_tensor
def _graph_prepare_batch(batch, device=None, non_blocking: bool = False):
    x, y = batch
    return (
        x.to(device),
        convert_tensor(y, device=device, non_blocking=non_blocking),
    )

def get_peak_mem_and_reset():
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.peak"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return peak_bytes_requirement / 1024 ** 3  # unit: GB

# Get forward pass information

ft = []
bt = []
fm = []
bm = []


with torch.no_grad():
    for i in tqdm(range(20)):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for batch in test_loader:

            x, y = _graph_prepare_batch(batch)
            out, sparsity, src_pe, graphs, attns = model(x.to('cuda'))

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        forward_t = start.elapsed_time(end) / 1000  # unit: milliseconds

        f_peak_mem = get_peak_mem_and_reset()
        ft.append(forward_t)
        fm.append(f_peak_mem)
    
    
    
# Get backward pass information

for i in tqdm(range(20)):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for batch in test_loader:

        x, y = _graph_prepare_batch(batch)
        out, sparsity, src_pe, graphs, attns = model(x.to('cuda'))
        out.mean().backward()

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()
    backward_t = start.elapsed_time(end) / 1000  # unit: milliseconds

    b_peak_mem = get_peak_mem_and_reset()
    bt.append(backward_t)
    bm.append(b_peak_mem)
    
print('EXP END \n')
print(ft)
print('\n')
print(fm)
print('\n')
print(bt)
print('\n')
print(bm)