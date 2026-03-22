"""
Training script for the Physics-Informed Graph Neural Network (PIGNN).

Usage:
    python training/train_pignn.py

Prerequisites:
    - Generate data first:  python data/create_data.py
    - Install deps:         pip install -r requirements.txt
"""

import os
import sys
import time

import numpy as np
import torch
from torch.nn import Softplus
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.pignn import PIGNN
from models.model_utility import (
    get_data_sets,
    convert_input_data,
    train,
    test_dev_set,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(0)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE       = 3
EPOCHS           = 1200
LR_INIT          = 8e-3
LR_FACTOR        = 0.5
LR_PATIENCE      = 1200
LR_THRESHOLD     = 1e-4
LR_MIN           = 1e-5
PINN             = True
ROLLOUT          = True
NOISE_LEVEL      = 0.0
GRADIENT_METHOD  = "direct"     # "direct" or "normalize"
HIDDEN           = 32
MSG_DIM          = 32
N_MP_LAYERS      = 2
EXP_NAME         = "pignn_bluerov2"
ALPHA_SMOOTH     = 0.6          # EMA smoothing for lr scheduler


def main():
    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, dev_loader, _, _ = get_data_sets(BATCH_SIZE)
    X0, U0, tc0, t0 = next(iter(train_loader))
    N_x  = X0.shape[-1]        # 9
    N_u  = U0.shape[-1]        # 4
    N_in = N_x + N_u + 1       # 14

    print(f"State dim:   {N_x}")
    print(f"Control dim: {N_u}")
    print(f"Input dim:   {N_in}")
    print(f"Device:      {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PIGNN(
        N_in=N_in,
        N_out=N_x,
        hidden=HIDDEN,
        msg_dim=MSG_DIM,
        n_mp_layers=N_MP_LAYERS,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LR_INIT)
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        threshold=LR_THRESHOLD,
        min_lr=LR_MIN,
    )

    # ------------------------------------------------------------------
    # TensorBoard
    # ------------------------------------------------------------------
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{EXP_NAME}_{timestamp}")

    # ------------------------------------------------------------------
    # Model save directory
    # ------------------------------------------------------------------
    model_dir = "models_saved"
    os.makedirs(model_dir, exist_ok=True)
    base_name = f"{EXP_NAME}_{GRADIENT_METHOD}"

    l_dev_best   = np.float32("inf")
    l_dev_smooth = 1.0

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    try:
        for epoch in trange(EPOCHS, desc="Training"):
            l_train = train(
                model, train_loader, optimizer, epoch, device, writer,
                pinn=PINN, rollout=ROLLOUT,
                noise_level=NOISE_LEVEL,
                gradient_method=GRADIENT_METHOD,
            )

            l_dev = test_dev_set(model, dev_loader, epoch, device, writer)

            l_dev_smooth = ALPHA_SMOOTH * l_dev_smooth + (1 - ALPHA_SMOOTH) * l_dev

            if l_dev < l_dev_best:
                best_path = os.path.join(
                    model_dir, f"{base_name}_best_dev_epoch_{epoch}"
                )
                torch.save(model.state_dict(), best_path)
                l_dev_best = l_dev

            scheduler.step(l_dev_smooth)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.flush()

        # Save final model
        final_path = os.path.join(model_dir, base_name)
        torch.save(model.state_dict(), final_path)

        writer.add_hparams(
            {
                "PINN":             PINN,
                "Rollout":          ROLLOUT,
                "lr_init":          LR_INIT,
                "lr_factor":        LR_FACTOR,
                "lr_patience":      LR_PATIENCE,
                "batch_size":       BATCH_SIZE,
                "epochs":           EPOCHS,
                "hidden":           HIDDEN,
                "msg_dim":          MSG_DIM,
                "n_mp_layers":      N_MP_LAYERS,
                "noise_level":      NOISE_LEVEL,
                "gradient_method":  GRADIENT_METHOD,
            },
            {
                "final_train_loss": l_train,
                "final_dev_loss":   l_dev,
            },
        )
        writer.close()

    except Exception as e:
        print(f"\nTraining error: {e}")
        writer.close()
        raise


if __name__ == "__main__":
    main()
