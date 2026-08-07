"""
Push the pretrained AraSSM encoder (MLM checkpoint) to the Hugging Face Hub.

This script does NOT modify src/models/mamba.py -- it wraps your existing
MambaForMaskedLM class in a subclass that adds Hub upload/download support,
so your actual codebase stays untouched.

Usage:
    pip install huggingface_hub --break-system-packages
    huggingface-cli login
    python push_to_hub.py
"""

import argparse

import torch
import yaml
from huggingface_hub import PyTorchModelHubMixin

from models.mamba import MambaForMaskedLM


class HubMambaForMaskedLM(MambaForMaskedLM, PyTorchModelHubMixin):
    """Same model, with push_to_hub()/from_pretrained() added via the mixin."""

    pass


def main(args):
    cfg = yaml.safe_load(open(args.config))
    mcfg = dict(cfg["model"])
    mcfg["vocab_size"] = args.vocab_size
    mcfg["pad_token_id"] = args.pad_token_id

    model = HubMambaForMaskedLM(mcfg)

    print(f"loading weights from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"WARNING -- missing keys: {missing}, unexpected keys: {unexpected}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model loaded OK: {n_params:,} parameters (~{n_params / 1e6:.1f}M)")

    print(f"pushing to hub as '{args.repo_id}' (private={args.private}) ...")
    model.push_to_hub(args.repo_id, private=args.private)
    print("model push complete.")

    if args.push_tokenizer:
        from transformers import AutoTokenizer

        print(f"pushing tokenizer '{args.tokenizer_name}' to the same repo ...")
        tok = AutoTokenizer.from_pretrained(args.tokenizer_name)
        tok.push_to_hub(args.repo_id, private=args.private)
        print("tokenizer push complete.")

    print(f"done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--checkpoint", default="checkpoint/latest.pt")
    p.add_argument("--repo_id", required=True, help="e.g. your-username/arassm-base")
    p.add_argument("--vocab_size", type=int, default=64000)
    p.add_argument("--pad_token_id", type=int, default=0)
    p.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="push as a private repo (default: True, safest option -- flip with --public)",
    )
    p.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="push as a PUBLIC repo instead of private",
    )
    p.add_argument(
        "--push_tokenizer",
        action="store_true",
        help="also push the AraBERTv02 tokenizer to the same repo",
    )
    p.add_argument("--tokenizer_name", default="aubmindlab/bert-base-arabertv02")
    args = p.parse_args()
    main(args)
