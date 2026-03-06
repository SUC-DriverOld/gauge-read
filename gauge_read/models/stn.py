import torch
import torch.nn as nn
import torchvision.models as models


class STNModel(nn.Module):
    def __init__(self, pretrained=True):
        super(STNModel, self).__init__()
        self.backbone = models.convnext_tiny(pretrained=pretrained)
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


class STNLoss(nn.Module):
    def __init__(self):
        super(STNLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.center_weight = 10.0

    def forward(self, pred_st, Minv_gt, pred_center, center_gt):
        """
        pred_st: [B, 9] 向量形式
        Minv_gt: [B, 3, 3] 真实矩阵
        pred_center: [B, 2] 预测的归一化圆心
        center_gt: [B, 2] 真实的归一化圆心
        """
        B = pred_st.shape[0]
        loss_reg = self.l1_loss(pred_st, Minv_gt.view(B, 9))
        loss_center = self.l1_loss(pred_center, center_gt)

        total_loss = loss_reg + self.center_weight * loss_center
        return total_loss, loss_reg, loss_center
