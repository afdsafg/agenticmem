"""GOAT-Bench evaluation entry point.

Usage:
    python run_goat_evaluation.py -cf cfg/eval_goat_server.yaml [--start 0.0] [--end 0.01]
"""
import argparse
import logging
import os
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("MPLBACKEND", "Agg")

from omegaconf import OmegaConf

from src.goat_runner import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", required=True, help="Path to eval_goat yaml")
    parser.add_argument("--start", type=float, default=0.0, help="Start ratio of episodes")
    parser.add_argument("--end", type=float, default=1.0, help="End ratio of episodes")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    results = main(cfg, start_ratio=args.start, end_ratio=args.end)
    print(f"\nDone. {sum(1 for r in results if r['success'])}/{len(results)} subtasks succeeded.")
