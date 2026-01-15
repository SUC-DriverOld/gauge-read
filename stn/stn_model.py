import torch
import torch.nn as nn
import torchvision.models as models


class STNModel(nn.Module):
    def __init__(self, pretrained=True):
        super(STNModel, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(2048, 8)

    def forward(self, x):
        pred_st = self.backbone(x)
        bsz = pred_st.shape[0]
        pred_st = torch.cat([pred_st, torch.ones(bsz, 1, device=pred_st.device)], dim=1)
        Minv_pred = pred_st.view(-1, 3, 3)
        return Minv_pred, pred_st


class STNLoss(nn.Module):
    def __init__(self):
        super(STNLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_st, Minv_gt):
        """
        pred_st: [B, 9] 向量形式
        Minv_gt: [B, 3, 3] 真实矩阵
        """
        B = pred_st.shape[0]
        loss_reg = self.l1_loss(pred_st, Minv_gt.view(B, 9))
        return loss_reg
