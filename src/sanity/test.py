import torch

from models.mamba import MambaForMaskedLM

torch.manual_seed(0)
device = "cuda"
config = {
    "vocab_size": 1000,
    "d_model": 64,
    "n_layer": 2,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "pad_token_id": 0,
    "dropout": 0.0,
}
model = MambaForMaskedLM(config).to(device).eval()

B, L = 2, 20
real_len = 10
input_ids_a = torch.randint(1, 1000, (B, L), device=device)
attn_mask = torch.zeros(B, L, device=device)
attn_mask[:, :real_len] = 1
input_ids_a[:, real_len:] = 0

input_ids_b = input_ids_a.clone()
input_ids_b[:, real_len:] = torch.randint(1, 1000, (B, L - real_len), device=device)

with torch.no_grad():
    out_a = model(input_ids_a, attn_mask)["logits"]
    out_b = model(input_ids_b, attn_mask)["logits"]

real_diff = (out_a[:, :real_len] - out_b[:, :real_len]).abs().max().item()
print("max diff at REAL positions from changing padding content:", real_diff)
