import argparse
import os
from pathlib import Path

from py_config_runner import ConfigObject

# from ax.service.managed_loop import optimize
from script import run_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Example application")
    parser.add_argument("--config", type=Path, help="Input configuration file")
    parser.add_argument("--use_hype_params", action="store_true")
    parser.add_argument("--data_type", type=str, default="")
    parser.add_argument("--exp_type", type=str, default="method")
    parser.add_argument("--g", type=str, default="")
    args = parser.parse_args()

    assert args.config is not None
    assert args.config.exists()

    config = ConfigObject(args.config)
    if args.g != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.g
        config.device = "cuda"
        config.g = args.g
        if len(args.g.split(",")) > 1:
            config.multi_gpu = True
            config.batch_size = config.batch_size * len(args.g.split(","))
        else:
            config.multi_gpu = False
    else:
        config.device = "cpu"
        config.multi_gpu = False

    print("start running")
    if args.exp_type == "summary":
        run_summary(config)
