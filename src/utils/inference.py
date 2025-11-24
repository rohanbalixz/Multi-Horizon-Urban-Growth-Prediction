"""
Inference and prediction utilities.

Handles:
- Model loading from checkpoint
- Autoregressive multi-horizon forecasting
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Loaded checkpoint with val_loss: {checkpoint.get('val_loss', 'N/A')}")
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
            # For dual-channel input, repeat prediction across channels
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


if __name__ == "__main__":
    print("Inference utilities loaded.")
