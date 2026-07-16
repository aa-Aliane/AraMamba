"""
Held-out evaluation for MLM pretraining: val loss, masked-token accuracy,
and perplexity. Kept separate from train.py so the eval logic can be run,
tested, or extended (e.g. later a downstream probe) independently of the
training loop.
"""

import math

import torch


@torch.no_grad()
def evaluate(model, val_loader, device, fp16):
    """Runs a full pass over val_loader and returns a dict of metrics:
      - loss: mean MLM cross-entropy loss over all masked positions
      - accuracy: top-1 accuracy of predictions at masked positions only
      - perplexity: exp(loss), the standard MLM perplexity metric

    Only meant to be called on the main rank (rank 0); this is a no_grad
    forward pass with no backward/allreduce, so DDP doesn't need every
    rank to participate.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct, total_masked = 0, 0

    for input_ids, attn_mask, labels in val_loader:
        input_ids, attn_mask, labels = (
            input_ids.to(device),
            attn_mask.to(device),
            labels.to(device),
        )
        with torch.amp.autocast(device_type="cuda", enabled=fp16):
            out = model(input_ids, attn_mask, labels)

        total_loss += out["loss"].item()
        n_batches += 1

        masked_positions = labels != -100
        if masked_positions.any():
            preds = out["logits"].argmax(dim=-1)
            correct += (
                (preds[masked_positions] == labels[masked_positions]).sum().item()
            )
            total_masked += masked_positions.sum().item()

    model.train()

    mean_loss = total_loss / max(1, n_batches)
    accuracy = correct / max(1, total_masked)
    # guard against overflow on a badly-diverged run
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")

    return {"loss": mean_loss, "accuracy": accuracy, "perplexity": perplexity}
