"""
Inference and prediction utilities.

Handles:
- Model loading from checkpoint
- Autoregressive multi-horizon forecasting
- MC Dropout uncertainty estimation
- Visualization of predictions
"""

import torch
import numpy as np
from pathlib import Path


def load_trained_model(model, checkpoint_path):
    """
    Load trained model from checkpoint.
    
    Args:
        model (nn.Module): Model instance
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        nn.Module: Loaded model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint with val_loss: {checkpoint.get('val_loss', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded state dict from {checkpoint_path}")
    model.eval()
    return model


def predict_next_timestep(model, input_sequence, device='cpu'):
    """
    Predict next timestep from input sequence.
    
    Args:
        model (nn.Module): Trained ConvLSTM model
        input_sequence (torch.Tensor): Shape (batch, seq_len, channels, H, W)
        device (str): Device for inference
        
    Returns:
        torch.Tensor: Predicted next timestep
    """
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.to(device)
        predictions, _ = model(input_sequence)
        # Return last prediction
        return predictions[:, -1, :, :, :]


def predict_future_step(model, input_sequence, device='cpu'):
    """
    Predict future step using autoregressive hidden state accumulation.
    Matches the notebook's forward(x, future_steps=1) logic.

    Args:
        model (nn.Module): Trained ConvLSTM model
        input_sequence (torch.Tensor): Shape (batch, seq_len, channels, H, W)
        device (str): Device for inference

    Returns:
        torch.Tensor: Predicted future timestep (batch, 1, H, W)
    """
    input_sequence = input_sequence.to(device)
    _, hidden_states = model(input_sequence)

    last_input = input_sequence[:, -1]
    x_t = last_input
    layer_hiddens = []
    for layer_idx, layer in enumerate(model.convlstm_layers):
        h, c = hidden_states[layer_idx]
        h, c = layer(x_t, (h, c))
        if hasattr(model, 'mc_dropouts'):
            h = model.mc_dropouts[layer_idx](h)
        hidden_states[layer_idx] = (h, c)
        x_t = h
        layer_hiddens.append(h)

    pred = model.decode(layer_hiddens)
    return pred


def mc_dropout_predict(model, input_sequence, n_passes=20, device='cpu'):
    """
    MC Dropout uncertainty estimation via multiple stochastic forward passes.
    
    Args:
        model (nn.Module): Model with MC Dropout
        input_sequence (torch.Tensor): Shape (batch, seq_len, channels, H, W)
        n_passes (int): Number of MC forward passes
        device (str): Device for inference
        
    Returns:
        mean (torch.Tensor): Mean prediction (batch, 1, H, W)
        std (torch.Tensor): Pixel-wise uncertainty (batch, 1, H, W)
        all_preds (torch.Tensor): All predictions (n_passes, batch, 1, H, W)
    """
    model.enable_mc_dropout()
    input_sequence = input_sequence.to(device)
    
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = predict_future_step(model, input_sequence, device)
            preds.append(pred.cpu())
    
    all_preds = torch.stack(preds, dim=0)  # (n_passes, B, 1, H, W)
    mean = all_preds.mean(dim=0)
    std = all_preds.std(dim=0)
    
    return mean, std, all_preds


def autoregressive_forecast(model, initial_sequence, num_steps, device='cpu'):
    """
    Generate multi-horizon forecast autoregressively.
    
    Args:
        model (nn.Module): Trained model
        initial_sequence (torch.Tensor): Initial observations (batch, seq_len, channels, H, W)
        num_steps (int): Number of future timesteps to forecast
        device (str): Device for inference
        
    Returns:
        list: List of predicted tensors for each future timestep
    """
    model.eval()
    forecasts = []
    current_sequence = initial_sequence.clone()
    
    with torch.no_grad():
        for step in range(num_steps):
            # Predict next step
            next_pred = predict_next_timestep(model, current_sequence, device)
            forecasts.append(next_pred.cpu())
            
            # Update sequence: remove oldest, append prediction
            # For multi-channel input, repeat prediction across channels
            next_input = next_pred.unsqueeze(1).repeat(1, 1, current_sequence.size(2), 1, 1)
            current_sequence = torch.cat([current_sequence[:, 1:, :, :, :], next_input], dim=1)
    
    return forecasts


def predict_tile(model, tile_data, checkpoint_path, device='cpu'):
    """
    Convenience function to predict on a single tile.
    
    Args:
        model (nn.Module): Model instance
        tile_data (numpy.ndarray): Tile data (seq_len, channels, H, W)
        checkpoint_path (str): Path to checkpoint
        device (str): Device for inference
        
    Returns:
        numpy.ndarray: Prediction for next timestep
    """
    model = load_trained_model(model, checkpoint_path)
    model = model.to(device)
    
    # Convert to tensor and add batch dimension
    tile_tensor = torch.from_numpy(tile_data).unsqueeze(0).float()
    
    prediction = predict_next_timestep(model, tile_tensor, device)
    return prediction.squeeze().cpu().numpy()


def predict_tile_with_uncertainty(model, tile_data, checkpoint_path,
                                  n_passes=20, device='cpu'):
    """
    Predict on a single tile with MC Dropout uncertainty.
    
    Args:
        model (nn.Module): Model instance with MC Dropout
        tile_data (numpy.ndarray): Tile data (seq_len, channels, H, W)
        checkpoint_path (str): Path to checkpoint
        n_passes (int): Number of MC forward passes
        device (str): Device for inference
        
    Returns:
        dict: {'mean': ndarray, 'std': ndarray, 'ci95_low': ndarray, 'ci95_high': ndarray}
    """
    model = load_trained_model(model, checkpoint_path)
    model = model.to(device)
    
    tile_tensor = torch.from_numpy(tile_data).unsqueeze(0).float()
    
    mean, std, all_preds = mc_dropout_predict(model, tile_tensor, n_passes, device)
    
    # Compute 95% CI
    all_np = all_preds.squeeze().numpy()  # (n_passes, H, W)
    ci95_low = np.percentile(all_np, 2.5, axis=0)
    ci95_high = np.percentile(all_np, 97.5, axis=0)
    
    return {
        'mean': mean.squeeze().numpy(),
        'std': std.squeeze().numpy(),
        'ci95_low': ci95_low,
        'ci95_high': ci95_high,
    }


if __name__ == "__main__":
    print("Inference utilities loaded.")
    print("Supports: deterministic prediction, MC Dropout uncertainty, autoregressive forecasting")
