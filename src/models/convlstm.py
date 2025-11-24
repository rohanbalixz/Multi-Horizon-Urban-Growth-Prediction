"""
ConvLSTM Model Architecture for Urban Growth Prediction

This module implements the dual-channel ConvLSTM architecture described in:
"Neural Time Capsule: Forecasting Urban Development Through Multi-Decadal 
Spatio-Temporal ConvLSTM with Built-Up and Road Network Inputs"

Reference: https://github.com/rohanbalixz/NeuralTimeCapsule
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell with spatial structure preservation.
    
    Args:
        input_channels (int): Number of input feature channels
        hidden_channels (int): Number of hidden state channels
        kernel_size (int): Size of convolutional kernel
    """
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        
        # Combined convolution for input, forget, cell, and output gates
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(self, x, states):
        h, c = states
        
        combined = torch.cat([x, h], dim=1)
        combined_conv = self.conv(combined)
        
        # Split into gate components
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)
        
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate
        
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for spatio-temporal sequence prediction.
    
    Architecture:
    - 2 ConvLSTM layers with 64 channels each
    - Dual-channel input: built-up density + road infrastructure
    - Autoregressive multi-horizon forecasting capability
    
    Args:
        input_channels (int): Number of input channels (default: 2)
        hidden_channels (int): Hidden state channels per layer (default: 64)
        num_layers (int): Number of ConvLSTM layers (default: 2)
        kernel_size (int): Convolutional kernel size (default: 3)
    """
    def __init__(self, input_channels=2, hidden_channels=64, num_layers=2, kernel_size=3):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Create multi-layer ConvLSTM stack
        layers = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            layers.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.layers = nn.ModuleList(layers)
        
        # Output projection: hidden state -> single-channel prediction
        self.output_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x, hidden_states=None):
        """
        Forward pass through ConvLSTM.
        
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, channels, height, width)
            hidden_states (list): Optional initial hidden states
            
        Returns:
            Tensor: Predictions of shape (batch, seq_len, 1, height, width)
        """
        batch_size, seq_len, _, height, width = x.size()
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                (torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device),
                 torch.zeros(batch_size, self.hidden_channels, height, width, device=x.device))
                for _ in range(self.num_layers)
            ]
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            
            # Pass through each ConvLSTM layer
            for layer_idx, layer in enumerate(self.layers):
                h, c = hidden_states[layer_idx]
                h_next, c_next = layer(x_t, (h, c))
                hidden_states[layer_idx] = (h_next, c_next)
                x_t = h_next  # Output of current layer -> input to next layer
            
            # Generate prediction from final hidden state
            output = self.output_conv(h_next)
            outputs.append(output)
        
        return torch.stack(outputs, dim=1), hidden_states
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(input_channels=2, hidden_channels=64, num_layers=2):
    """
    Factory function to create ConvLSTM model with standard configuration.
    
    Args:
        input_channels (int): Number of input channels (default: 2)
        hidden_channels (int): Hidden channels per layer (default: 64)
        num_layers (int): Number of ConvLSTM layers (default: 2)
        
    Returns:
        ConvLSTM: Initialized model instance
    """
    model = ConvLSTM(input_channels, hidden_channels, num_layers)
    print(f"Created ConvLSTM with {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    
    # Test forward pass
    batch_size, seq_len, height, width = 4, 3, 128, 128
    x = torch.randn(batch_size, seq_len, 2, height, width)
    
    predictions, _ = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Model parameters: {model.count_parameters():,}")
