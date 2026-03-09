import torch
import numpy as np


def _param2theta_batch(param, w, h):
    bsz = param.shape[0]
    ones_row = torch.tensor([0.0, 0.0, 1.0], dtype=param.dtype, device=param.device).view(1, 1, 3).repeat(bsz, 1, 1)
    param3 = torch.cat([param, ones_row], dim=1)
    param_inv = torch.linalg.inv(param3)

    theta = torch.zeros((bsz, 2, 3), dtype=param.dtype, device=param.device)
    theta[:, 0, 0] = param_inv[:, 0, 0]
    theta[:, 0, 1] = param_inv[:, 0, 1] * h / w
    theta[:, 0, 2] = param_inv[:, 0, 2] * 2 / w + theta[:, 0, 0] + theta[:, 0, 1] - 1
    theta[:, 1, 0] = param_inv[:, 1, 0] * w / h
    theta[:, 1, 1] = param_inv[:, 1, 1]
    theta[:, 1, 2] = param_inv[:, 1, 2] * 2 / h + theta[:, 1, 0] + theta[:, 1, 1] - 1
    return theta


def param2theta(param, w, h):
    param = np.vstack([param, [0, 0, 1]])
    param = np.linalg.inv(param)

    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta


def batch_roi_transform(feature_map, boxes, mapping, size=(32, 180)):
    resize_h, resize_w = size
    if len(boxes) == 0:
        ch = 1 if feature_map.shape[1] == 3 else feature_map.shape[1]
        return torch.empty((0, ch, resize_h, resize_w), device=feature_map.device, dtype=feature_map.dtype)

    device = feature_map.device
    dtype = feature_map.dtype
    mapping_t = torch.as_tensor(mapping, device=device, dtype=torch.long)
    boxes_t = torch.as_tensor(boxes, device=device, dtype=dtype).view(-1, 8)

    feats = feature_map[mapping_t]
    bsz, c, h, w = feats.shape

    # src points: p1(x1,y1), p2(x2,y2), p4(x4,y4)
    src = torch.stack(
        [
            boxes_t[:, [0, 1]],
            boxes_t[:, [2, 3]],
            boxes_t[:, [6, 7]],
        ],
        dim=1,
    )
    dst = torch.tensor([[0.0, 0.0], [float(resize_w), 0.0], [0.0, float(resize_h)]], device=device, dtype=dtype)
    dst = dst.unsqueeze(0).repeat(bsz, 1, 1)

    src_homo = torch.cat([src, torch.ones((bsz, 3, 1), device=device, dtype=dtype)], dim=2)
    # Solve src_homo @ A^T = dst  => A = (solve result)^T, where A maps src -> dst
    a_t = torch.linalg.solve(src_homo, dst)
    param = a_t.transpose(1, 2)
    theta = _param2theta_batch(param, w, h)

    grid = torch.nn.functional.affine_grid(theta, feats.size(), align_corners=True)
    feat_rot = torch.nn.functional.grid_sample(feats, grid, align_corners=True)
    rois = feat_rot[:, :, 0:resize_h, 0:resize_w]

    if c == 3:
        gray = (0.2989 * rois[:, 0] + 0.5870 * rois[:, 1] + 0.1140 * rois[:, 2]).unsqueeze(1)
        return gray.to(rois.dtype)
    return rois