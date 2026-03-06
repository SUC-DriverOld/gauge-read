import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class STNModel(nn.Module):
    def __init__(self, pretrained=True):
        super(STNModel, self).__init__()
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, 10)

    def forward(self, x):
        out = self.backbone(x)
        pred_st = out[:, :8]
        pred_center = out[:, 8:]

        bsz = pred_st.shape[0]
        pred_st = torch.cat([pred_st, torch.ones(bsz, 1, device=pred_st.device)], dim=1)
        Minv_pred = pred_st.view(-1, 3, 3)
        return Minv_pred, pred_st, pred_center
