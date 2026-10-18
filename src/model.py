import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBackbone(nn.Module):
    def __init__(self, in_ch: int, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base, 11, padding=5),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base, base*2, 9, padding=4),
            nn.BatchNorm1d(base*2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(base*2, base*4, 7, padding=3),
            nn.BatchNorm1d(base*4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = base*4

    def forward(self, x):
        # x: [B,C,T]
        h = self.net(x).squeeze(-1)
        return h

class MTLNet(nn.Module):
    def __init__(self, in_ch: int, num_labels: int = 3, reg_dim: int = 7):
        super().__init__()
        self.backbone = CNNBackbone(in_ch)
        self.cls = nn.Linear(self.backbone.out_dim, num_labels)
        self.reg = nn.Linear(self.backbone.out_dim, reg_dim)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.cls(h)
        reg = self.reg(h)
        return logits, reg

def masked_reg_loss(pred, target, y_multi):
    """Conditional/masked regression:
    - unbalance scalar active if y_multi[:,0]==1
    - misalign xyz active if y_multi[:,1]==1
    - rub onehot active if y_multi[:,2]==1
    """
    device = pred.device
    m = torch.zeros_like(pred, device=device)
    m[:,0] = y_multi[:,0]
    m[:,1:4] = y_multi[:,1].unsqueeze(1)
    m[:,4:7] = y_multi[:,2].unsqueeze(1)
    diff = (pred - target) * m
    denom = m.sum().clamp_min(1.0)
    return (diff.pow(2).sum() / denom)
import torch
import torch.nn.functional as F

def masked_scaled_huber_loss(pred, target, y_multi, scale=None, beta=1.0):
    """
    pred/target: [B,7]
    y_multi: [B,3]
    scale: [7]  每一维的尺度(建议用std或range)，用于单位归一化
    beta: SmoothL1 的转折点，越小越接近L1，越大越接近L2
    """
    device = pred.device
    m = torch.zeros_like(pred, device=device)
    m[:, 0] = y_multi[:, 0]
    m[:, 1:4] = y_multi[:, 1].unsqueeze(1)
    m[:, 4:7] = y_multi[:, 2].unsqueeze(1)

    diff = pred - target
    if scale is not None:
        sc = torch.as_tensor(scale, device=device, dtype=diff.dtype).view(1, -1)
        sc = torch.clamp(sc, min=1e-6)
        diff = diff / sc

    # SmoothL1 on normalized diff
    loss_mat = F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=beta, reduction="none")
    loss_mat = loss_mat * m
    denom = m.sum().clamp_min(1.0)
    return loss_mat.sum() / denom
