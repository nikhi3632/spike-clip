import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import LIFNode

class CoarseSNN(nn.Module):
    """
    Optimized Coarse Reconstruction SNN.
    Input: [B, T, H, W]
    Output: [B, 3, H, W]
    """

    def __init__(self, in_channels=1, out_channels=3, time_steps=200, threshold=1.0, decay=2.0):
        super().__init__()
        self.time_steps = time_steps

        # Temporal + spatial conv layers (stepwise)
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.lif1 = LIFNode(tau=decay, v_threshold=threshold)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.lif2 = LIFNode(tau=decay, v_threshold=threshold)

        self.conv3 = nn.Conv2d(32, out_channels, 3, padding=1)
        self.lif3 = LIFNode(tau=decay, v_threshold=threshold)

    def forward(self, x):
        """
        x: [B, T, H, W]
        """
        B, T, H, W = x.shape
        x = x.unsqueeze(1)  # [B, 1, T, H, W]

        self.lif1.reset()
        self.lif2.reset()
        self.lif3.reset()

        out = 0
        for t in range(T):
            xt = x[:, :, t, :, :]  # [B, 1, H, W]
            h = self.lif1(self.conv1(xt))
            h = self.lif2(self.conv2(h))
            h = self.lif3(self.conv3(h))
            out += h

        out = out / T
        out = torch.sigmoid(out)  # normalize to [0,1]
        return out
