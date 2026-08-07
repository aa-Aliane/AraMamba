import yaml

from models.mamba import MambaEncoder

cfg = yaml.safe_load(open("configs/base.yaml"))
mcfg = dict(cfg["model"])
mcfg["vocab_size"] = 64000  # AraBERTv02 tokenizer vocab size
mcfg["pad_token_id"] = 0

encoder = MambaEncoder(mcfg)
n_params = sum(p.numel() for p in encoder.parameters())
print(f"AraSSM encoder: {n_params:,} parameters (~{n_params / 1e6:.1f}M)")
