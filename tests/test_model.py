"""
Unit tests for ConvLSTM model and baseline architectures.
"""

import torch
import numpy as np
import pytest
from src.models.convlstm import ConvLSTM, ConvLSTMCell, create_model


def test_convlstm_cell_forward():
    """Test ConvLSTMCell forward pass."""
    batch_size, channels, height, width = 2, 2, 32, 32
    hidden_channels = 16
    
    cell = ConvLSTMCell(channels, hidden_channels)
    x = torch.randn(batch_size, channels, height, width)
    h = torch.zeros(batch_size, hidden_channels, height, width)
    c = torch.zeros(batch_size, hidden_channels, height, width)
    
    h_next, c_next = cell(x, (h, c))
    
    assert h_next.shape == (batch_size, hidden_channels, height, width)
    assert c_next.shape == (batch_size, hidden_channels, height, width)


def test_convlstm_forward():
    """Test ConvLSTM forward pass."""
    batch_size, seq_len, channels, height, width = 2, 3, 2, 64, 64
    hidden_channels = 32
    num_layers = 2
    
    model = ConvLSTM(channels, hidden_channels, num_layers)
    x = torch.randn(batch_size, seq_len, channels, height, width)
    
    predictions, hidden_states = model(x)
    
    # Check output shape
    assert predictions.shape == (batch_size, seq_len, 1, height, width)
    
    # Check hidden states
    assert len(hidden_states) == num_layers
    for h, c in hidden_states:
        assert h.shape == (batch_size, hidden_channels, height, width)
        assert c.shape == (batch_size, hidden_channels, height, width)


def test_convlstm_output_range():
    """Verify sigmoid decoder constrains outputs to [0, 1]."""
    model = create_model(input_channels=2, hidden_channels=64, num_layers=2)
    model.eval()
    x = torch.randn(1, 2, 2, 32, 32)
    with torch.no_grad():
        preds, _ = model(x)
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_create_model():
    """Test model factory function."""
    model = create_model(input_channels=2, hidden_channels=64, num_layers=2)
    assert isinstance(model, ConvLSTM)
    assert model.count_parameters() > 0


def test_parameter_count():
    """Test parameter counting for 3-channel model with skip-connection decoder."""
    model = create_model(input_channels=3, hidden_channels=64, num_layers=2)
    param_count = model.count_parameters()
    # skip_proj adds 64*2 * 64 + 64 = 8256 params over old 472,897
    assert param_count == 481153, f"Expected 481,153 but got {param_count:,}"


def test_skip_connection_decode():
    """Test skip-connection decoder fuses all layer hidden states."""
    model = create_model(input_channels=3, hidden_channels=64, num_layers=2)
    model.eval()
    h0 = torch.randn(1, 64, 32, 32)
    h1 = torch.randn(1, 64, 32, 32)
    with torch.no_grad():
        out = model.decode([h0, h1])
    assert out.shape == (1, 1, 32, 32)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_autoregressive_prediction():
    """Test multi-step autoregressive prediction."""
    model = create_model()
    model.eval()
    
    batch_size, seq_len = 1, 3
    x = torch.randn(batch_size, seq_len, 2, 128, 128)
    
    with torch.no_grad():
        predictions, _ = model(x)
    
    assert predictions.shape == (batch_size, seq_len, 1, 128, 128)
    assert not torch.isnan(predictions).any()




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
