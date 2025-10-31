import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    """
    Stage 1: Coarse reconstruction loss.
    Supports L1 (MAE) or L2 (MSE) loss.
    """
    def __init__(self, loss_type="l1"):
        super().__init__()
        loss_type = loss_type.lower()
        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predicted coarse image [B, C, H, W]
            target: ground truth image [B, C, H, W]
        Returns:
            Loss scalar
        """
        return self.criterion(pred, target)


# Optional: you can add future losses here, e.g. CLIP, perceptual, refinement
class DummyLoss(nn.Module):
    """Placeholder for Stage 2/3 loss"""
    def forward(self, pred, target):
        return torch.tensor(0.0, device=pred.device)
