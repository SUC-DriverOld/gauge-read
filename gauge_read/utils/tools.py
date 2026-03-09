import numpy as np
import cv2
import torch
from kornia.geometry.transform import warp_perspective


_warp_norm_cache = {}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_device(*tensors, device):
    if len(tensors) < 2:
        return tensors[0].to(device)
    return tuple(t.to(device) for t in tensors)


def _get_warp_norm_mats(device, dtype, sz):
    key = (str(device), str(dtype), int(sz))
    if key not in _warp_norm_cache:
        s, t = sz / 2.0, 1.0
        left = torch.tensor([[s, 0, t * s], [0, s, t * s], [0, 0, 1]], device=device, dtype=dtype)
        right = torch.tensor([[1 / s, 0, -t], [0, 1 / s, -t], [0, 0, 1]], device=device, dtype=dtype)
        _warp_norm_cache[key] = (left, right)
    return _warp_norm_cache[key]


def collate_fn(batch):
    img, pointer_mask, dail_mask, text_mask, train_mask, boxes, transcripts = zip(*batch)
    bs = len(img)
    images = []
    pointer_maps = []
    dail_maps = []
    text_maps = []
    training_masks = []

    for i in range(bs):
        if img[i] is not None:
            a = torch.from_numpy(img[i]).float()
            images.append(a)
            b = torch.from_numpy(pointer_mask[i]).long()
            pointer_maps.append(b)
            c = torch.from_numpy(dail_mask[i]).long()
            dail_maps.append(c)
            d = torch.from_numpy(text_mask[i]).long()
            text_maps.append(d)
            e = torch.from_numpy(train_mask[i])
            training_masks.append(e)

    images = torch.stack(images, 0)
    pointer_maps = torch.stack(pointer_maps, 0)
    dail_maps = torch.stack(dail_maps, 0)
    text_maps = torch.stack(text_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    mapping = []
    texts = []
    bboxs = []
    for index, gt in enumerate(zip(transcripts, boxes)):
        for t, b in zip(gt[0], gt[1]):
            mapping.append(index)
            texts.append(t)
            bboxs.append(b)

    mapping = np.array(mapping)
    texts = np.array(texts)
    bboxs = np.stack(bboxs, axis=0)
    bboxs = np.concatenate([bboxs, np.ones((len(bboxs), 1))], axis=1).astype(np.float32)

    return images, pointer_maps, dail_maps, text_maps, training_masks, texts, bboxs, mapping


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0, 1] != leftMost[1, 1]:
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    else:
        leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
    (tl, bl) = leftMost
    if rightMost[0, 1] != rightMost[1, 1]:
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    else:
        rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
    (tr, br) = rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl])


def warp(img, Minv_pred, device=None, sz=224):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = Minv_pred.dtype if torch.is_tensor(Minv_pred) else torch.float32
    left, right = _get_warp_norm_mats(device, dtype, sz)
    Minv_pred = left @ Minv_pred.to(device=device, dtype=dtype) @ right
    img_ = warp_perspective(img, Minv_pred, (sz, sz))
    return img_


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
    dtype = Minv_pred.dtype if torch.is_tensor(Minv_pred) else points.dtype
    left, right = _get_warp_norm_mats(device, dtype, sz)

    # 构建与 warp() 一致的最终变换矩阵
    transform_matrix = left @ Minv_pred.to(device=device, dtype=dtype) @ right  # [B, 3, 3]

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
