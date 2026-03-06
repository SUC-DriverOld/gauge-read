import torch
from kornia.geometry.transform import warp_perspective
import cv2
import numpy as np


def warp_points(points, Minv_pred, device=None, sz=224):
    """
    points: [B, 2] 或者是 [B, N, 2] 的 torch.Tensor，范围在图像像素坐标 [0, sz) 或者由外界预处理
    Minv_pred: [B, 3, 3]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if points.dim() == 2:
        # [B, 2] -> [B, 1, 2]
        points = points.unsqueeze(1)

    B, N, _ = points.shape
    s, t = sz / 2.0, 1.0

    # 构建与 warp() 一致的最终变换矩阵
    transform_matrix = (
        torch.Tensor([[s, 0, t * s], [0, s, t * s], [0, 0, 1]]).to(device)
        @ Minv_pred.to(device)
        @ torch.Tensor([[1 / s, 0, -t], [0, 1 / s, -t], [0, 0, 1]]).to(device)
    )  # [B, 3, 3]

    # 转换为齐次坐标 [B, N, 3]
    points_h = torch.cat([points, torch.ones(B, N, 1, device=device)], dim=-1)

    # 矩阵乘法
    # transform_matrix: [B, 3, 3]
    # points_h.transpose(1, 2): [B, 3, N]
    warped_points_h = torch.bmm(transform_matrix, points_h.transpose(1, 2))  # [B, 3, N]
    warped_points_h = warped_points_h.transpose(1, 2)  # [B, N, 3]

    # 归一化
    warped_points = warped_points_h[..., :2] / (warped_points_h[..., 2:3] + 1e-8)

    return warped_points.squeeze(1)


def draw_points_on_batch(img_batch, points, radius=3, color=(1.0, 0.0, 0.0)):
    """
    img_batch: [B, C, H, W] 范围在0-1的tensor
    points: [B, 2] 对应于 [x, y] 像素坐标
    """
    img_np = img_batch.detach().cpu().numpy()
    pts_np = points.detach().cpu().numpy()

    B, C, H, W = img_np.shape
    out_imgs = []
    for i in range(B):
        # 转换到 [H, W, C] 进行 OpenCV 操作
        img_single = img_np[i].transpose(1, 2, 0).copy()

        x, y = int(pts_np[i, 0]), int(pts_np[i, 1])
        if 0 <= x < W and 0 <= y < H:
            # 绘制红点
            cv2.circle(img_single, (x, y), radius, color, -1)

        # 再转回来 [C, H, W]
        out_imgs.append(img_single.transpose(2, 0, 1))

    return torch.from_numpy(np.stack(out_imgs)).to(img_batch.device)


def warp(img, Minv_pred, device=None, sz=224):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    s, t = sz / 2.0, 1.0
    Minv_pred = (
        torch.Tensor([[s, 0, t * s], [0, s, t * s], [0, 0, 1]]).to(device)
        @ Minv_pred.to(device)
        @ torch.Tensor([[1 / s, 0, -t], [0, 1 / s, -t], [0, 0, 1]]).to(device)
    )
    img_ = warp_perspective(img, Minv_pred, (sz, sz))
    return img_
