import torch, sys, numpy as np
sys.path.insert(0, '.')
from src.models.convlstm import ConvLSTM3Channel
import torch.nn as nn

# 1. Load checkpoint into src model
ckpt = torch.load('models/best_3ch_mc_model.pth', map_location='cpu', weights_only=True)
state = ckpt.get('model_state_dict', ckpt)
print("Checkpoint keys (first 4):", list(state.keys())[:4])

src = ConvLSTM3Channel()
src.load_state_dict(state)
src.eval()
print(f"src model: {sum(p.numel() for p in src.parameters()):,} params")

# 2. Load into inline ConvLSTMModel
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size // 2)
    def forward(self, x, states):
        h, c = states
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=64, num_layers=2, mc_dropout=0.1):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        layers = [ConvLSTMCell(input_channels if i == 0 else hidden_channels, hidden_channels) for i in range(num_layers)]
        self.convlstm_layers = nn.ModuleList(layers)
        self.mc_dropouts = nn.ModuleList([nn.Dropout2d(p=mc_dropout) for _ in range(num_layers)])
        self.skip_proj = nn.Conv2d(hidden_channels * num_layers, hidden_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        B, T, C, H, W = x.shape
        states = [(torch.zeros(B, self.hidden_channels, H, W), torch.zeros(B, self.hidden_channels, H, W)) for _ in range(self.num_layers)]
        for t in range(T):
            x_t = x[:, t]
            hiddens = []
            for i, layer in enumerate(self.convlstm_layers):
                h, c = states[i]
                h, c = layer(x_t, (h, c))
                h = self.mc_dropouts[i](h)
                states[i] = (h, c)
                x_t = h
                hiddens.append(h)
        fused = self.skip_proj(torch.cat(hiddens, dim=1))
        return self.decoder(fused)

inline = ConvLSTMModel()
result = inline.load_state_dict(state, strict=True)
inline.eval()
print(f"inline model: {sum(p.numel() for p in inline.parameters()):,} params")
print(f"load_state_dict: {result}")

# 3. Numerical equivalence check (dropout off since eval)
x = torch.randn(1, 8, 3, 16, 16)
with torch.no_grad():
    out_src    = src(x)[0][:, -1, 0]       # src returns (outputs, states); last timestep
    out_inline = inline(x)[:, 0]            # inline returns (B, 1, H, W)

diff = (out_src - out_inline).abs().max().item()
print(f"Max abs output difference (src vs inline): {diff:.2e}")
print("PASS" if diff < 1e-5 else f"MISMATCH — diff={diff}")
