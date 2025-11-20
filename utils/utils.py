import torch
from kornia.geometry.transform import warp_perspective


def warp(img, Minv_pred, device=None, sz=224):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    s, t = sz / 2.0, 1.0
    Minv_pred = (
        torch.Tensor([[s, 0, t * s], [0, s, t * s], [0, 0, 1]]).to(device)
        @ Minv_pred
        @ torch.Tensor([[1 / s, 0, -t], [0, 1 / s, -t], [0, 0, 1]]).to(device)
    )
    img_ = warp_perspective(img, Minv_pred, (sz, sz))
    return img_
