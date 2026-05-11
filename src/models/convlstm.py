"""
ConvLSTM Model Architecture for Urban Growth Prediction

This module implements the multi-channel ConvLSTM architecture described in:
"Neural Time Capsule: Forecasting Urban Development Through Multi-Decadal
Spatio-Temporal ConvLSTM with Built-Up, Volume, and Population Inputs"

Supports:
  - 3-channel input (built-up + volume + population)
  - Optional MC Dropout for Bayesian uncertainty estimation

Reference: https://github.com/rohanbalixz/NeuralTimeCapsule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    - Multi-channel input: built-up + volume + population
    - MC Dropout (p=0.1) after each ConvLSTM layer for uncertainty
    - Skip-connection decoder: fuses hidden states from ALL ConvLSTM layers
      via 1x1 projection, then decodes (64 -> 32 -> 16 -> 1)
    - Autoregressive multi-horizon forecasting capability
    
    Args:
        input_channels (int): Number of input channels (default: 2; use 3 for population)
        hidden_channels (int): Hidden state channels per layer (default: 64)
        num_layers (int): Number of ConvLSTM layers (default: 2)
        kernel_size (int): Convolutional kernel size (default: 3)
        mc_dropout (float): MC Dropout probability after each ConvLSTM layer (default: 0.0)
    """
    def __init__(self, input_channels=2, hidden_channels=64, num_layers=2,
                 kernel_size=3, mc_dropout=0.0):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.mc_dropout_p = mc_dropout
        
        # Create multi-layer ConvLSTM stack
        # NOTE: attribute named 'convlstm_layers' to match trained checkpoint keys
        convlstm_layers = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            convlstm_layers.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))
        self.convlstm_layers = nn.ModuleList(convlstm_layers)
        
        # MC Dropout layers (one per ConvLSTM layer)
        # These remain active during inference when enable_mc_dropout() is called
        self.mc_dropouts = nn.ModuleList([
            nn.Dropout2d(p=mc_dropout) for _ in range(num_layers)
        ])

        # Skip-connection projection: fuse hidden states from ALL ConvLSTM layers
        # This lets the decoder see both low-level (early layer) and high-level
        # (final layer) spatial features, similar to U-Net skip connections.
        self.skip_proj = nn.Conv2d(
            hidden_channels * num_layers, hidden_channels, kernel_size=1
        )

        # Decoder: progressive channel reduction 64 -> 32 -> 16 -> 1
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] to match normalized density targets
        )
    
    def decode(self, layer_hiddens):
        """
        Decode with skip connections from all ConvLSTM layers.

        Args:
            layer_hiddens (list[Tensor]): Hidden state from each layer,
                each of shape (batch, hidden_channels, H, W)

        Returns:
            Tensor: Prediction (batch, 1, H, W)
        """
        fused = self.skip_proj(torch.cat(layer_hiddens, dim=1))
        return self.decoder(fused)

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

            # Pass through each ConvLSTM layer with MC Dropout
            layer_hiddens = []
            for layer_idx, layer in enumerate(self.convlstm_layers):
                h, c = hidden_states[layer_idx]
                h_next, c_next = layer(x_t, (h, c))
                # Apply MC Dropout to hidden state output
                h_next = self.mc_dropouts[layer_idx](h_next)
                hidden_states[layer_idx] = (h_next, c_next)
                x_t = h_next  # Output of current layer -> input to next layer
                layer_hiddens.append(h_next)

            # Decode with skip connections from all layers
            output = self.decode(layer_hiddens)
            outputs.append(output)

        return torch.stack(outputs, dim=1), hidden_states
    
    def enable_mc_dropout(self):
        """
        Enable MC Dropout for inference (keeps dropout active during eval).
        Call this before running MC forward passes for uncertainty estimation.
        """
        self.eval()
        for drop in self.mc_dropouts:
            drop.train()
    
    def mc_predict(self, x, n_passes=20, hidden_states=None):
        """
        Run multiple stochastic forward passes for MC Dropout uncertainty.
        
        Args:
            x (Tensor): Input (batch, seq_len, channels, H, W)
            n_passes (int): Number of MC forward passes (default: 20)
            hidden_states: Optional initial states
            
        Returns:
            mean (Tensor): Mean prediction (batch, seq_len, 1, H, W)
            std (Tensor): Pixel-wise uncertainty (batch, seq_len, 1, H, W)
            all_preds (Tensor): All pass predictions (n_passes, batch, seq_len, 1, H, W)
        """
        self.enable_mc_dropout()
        
        all_preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds, _ = self.forward(x, hidden_states)
                all_preds.append(preds)
        
        all_preds = torch.stack(all_preds, dim=0)  # (n_passes, B, T, 1, H, W)
        mean = all_preds.mean(dim=0)
        std = all_preds.std(dim=0)
        
        return mean, std, all_preds
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConvLSTM3Channel(ConvLSTM):
    """
    ConvLSTM with 3 input channels (built-up + volume + population density)
    and MC Dropout enabled by default.

    Input shape: [batch, time, 3, 128, 128]
    """
    def __init__(self, hidden_channels=64, num_layers=2, kernel_size=3,
                 mc_dropout=0.1):
        super().__init__(
            input_channels=3,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            mc_dropout=mc_dropout,
        )


def create_model(input_channels=2, hidden_channels=64, num_layers=2, mc_dropout=0.0):
    """
    Factory function to create ConvLSTM model with standard configuration.
    
    Args:
        input_channels (int): Number of input channels (default: 2; use 3 for population)
        hidden_channels (int): Hidden channels per layer (default: 64)
        num_layers (int): Number of ConvLSTM layers (default: 2)
        mc_dropout (float): MC Dropout probability (default: 0.0; use 0.1 for uncertainty)
        
    Returns:
        ConvLSTM: Initialized model instance
    """
    model = ConvLSTM(input_channels, hidden_channels, num_layers, mc_dropout=mc_dropout)
    print(f"Created ConvLSTM with {model.count_parameters():,} parameters"
          f" (input_channels={input_channels}, mc_dropout={mc_dropout})")
    return model


if __name__ == "__main__":
    # Test 2-channel model
    print("--- 2-channel model ---")
    model_2ch = create_model(input_channels=2)
    x2 = torch.randn(4, 3, 2, 128, 128)
    preds2, _ = model_2ch(x2)
    print(f"Input: {x2.shape} -> Output: {preds2.shape}")
    print(f"Parameters: {model_2ch.count_parameters():,}")

    # Test 3-channel model with MC Dropout + skip connections
    print("\n--- 3-channel model + MC Dropout + skip decoder ---")
    model_3ch = create_model(input_channels=3, mc_dropout=0.1)
    x3 = torch.randn(4, 2, 3, 128, 128)
    preds3, _ = model_3ch(x3)
    print(f"Input: {x3.shape} -> Output: {preds3.shape}")
    print(f"Parameters: {model_3ch.count_parameters():,}")

    # Test decode() with explicit layer hiddens
    print("\n--- Skip-connection decode() test ---")
    h0 = torch.randn(4, 64, 128, 128)
    h1 = torch.randn(4, 64, 128, 128)
    out = model_3ch.decode([h0, h1])
    print(f"decode([h0, h1]) -> {out.shape}")

    # Test MC Dropout uncertainty estimation
    print("\n--- MC Dropout uncertainty (20 passes) ---")
    mean, std, all_preds = model_3ch.mc_predict(x3, n_passes=20)
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Mean uncertainty: {std.mean():.6f}")
    print(f"Max uncertainty: {std.max():.6f}")

    # Test ConvLSTM3Channel convenience class
    print("\n--- ConvLSTM3Channel convenience class ---")
    model_3ch_v2 = ConvLSTM3Channel()
    print(f"Parameters: {model_3ch_v2.count_parameters():,}")
    print(f"MC Dropout p: {model_3ch_v2.mc_dropout_p}")
