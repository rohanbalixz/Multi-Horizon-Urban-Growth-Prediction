"""
Full-validation-set evaluation utilities.

Computes MSE, MAE, and RMSE across **all** held-out validation tiles
rather than a single sample, ensuring the numbers reported in the paper
are reproducible.
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def evaluate_model_on_loader(model, val_loader, device="cpu"):
    """
    Compute MSE, MAE, and RMSE over an entire DataLoader.

    Args:
        model (nn.Module): Trained model (must accept (B, T, C, H, W) input).
        val_loader (DataLoader): Validation set loader.
        device (str): 'cpu' or 'cuda'.

    Returns:
        dict with keys 'mse', 'mae', 'rmse', 'n_tiles'.
    """
    model.eval()
    mse_criterion = nn.MSELoss(reduction="sum")

    total_se = 0.0     # sum of squared errors
    total_ae = 0.0     # sum of absolute errors
    total_pixels = 0   # total number of scalar predictions

    with torch.no_grad():
        for sequences, targets in tqdm(val_loader, desc="Evaluating"):
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Model may return (predictions, hidden_states) or just predictions
            output = model(sequences, future_steps=1) if hasattr(model, "convlstm_layers") else model(sequences)
            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output

            # Align shapes: predictions may be (B, T_out, 1, H, W)
            if predictions.dim() == 5 and predictions.size(1) == 1:
                predictions = predictions[:, 0]  # (B, 1, H, W)

            n = predictions.numel()
            total_se += ((predictions - targets) ** 2).sum().item()
            total_ae += (predictions - targets).abs().sum().item()
            total_pixels += n

    mse = total_se / total_pixels
    mae = total_ae / total_pixels
    rmse = np.sqrt(mse)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "n_tiles": len(val_loader.dataset),
    }

