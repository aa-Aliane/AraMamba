# Pilot Experiment Log

## 1. Overfitting Check (500 steps)
*Set: dropout: 0.0, weight_decay: 0.0, max_steps: 500*

| Step | Loss | Target |
| :--- | :--- | :--- |
| 1    |      | ~10.0+ |
| 100  |      | |
| 250  |      | |
| 500  |      | < 4.0 (Success) |

---

## 2. VRAM stress test (100 steps)
*Target: Keep Peak VRAM < 10.0 GB*

| Run | Batch Size | Grad Accum | Peak VRAM | OOM? (Y/N) | t/s (Speed) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A | 16 | 4 | | | |
| B | 12 | 6 | | | |
| C | 8 | 8 | | | |

---

## 3. LR Trajectory Sweep (2000 steps)
*Set: dropout: 0.1, weight_decay: 0.01*

### Run A (lr: 5e-4)
* Step 500 Loss: 
* Step 1000 Loss: 
* Step 2000 Loss: 
* Behavior (Smooth / Spiky / NaN): 

### Run B (lr: 3e-4)
* Step 500 Loss: 
* Step 1000 Loss: 
* Step 2000 Loss: 
* Behavior (Smooth / Spiky / NaN): 

---

## Final Production Decision
* **Batch Size (per GPU):** * **Grad Accum Steps:** * **Learning Rate:** ```
