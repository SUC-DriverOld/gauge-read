import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2
import numpy as np
from argparse import ArgumentParser
from utils import warp, warp_points
from stn_model import STNModel


def load_stn_model(model_path, device="cuda"):
    model_stn = STNModel(pretrained=False)
    model_stn.load_state_dict(torch.load(model_path, map_location=device))
    model_stn.to(device)
    model_stn.eval()
    return model_stn


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.transpose(2, 0, 1) / 255.0  # (H, W, C) -> (C, H, W)
    img = torch.Tensor(img).unsqueeze(0)  # 添加batch维度
    return img


def postprocess_image(img_tensor):
    img = img_tensor.squeeze(0).cpu().numpy()
    img = img.transpose(1, 2, 0)  # (c, h, w) -> (h, w, c)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def rectify_clock_image(model_stn, image_path, device="cuda"):
    img = preprocess_image(image_path)
    img = img.float().to(device)

    with torch.no_grad():
        Minv_pred, pred_st, pred_center = model_stn(img)
        img_warped = warp(img, Minv_pred)
        
        # 将归一化圆心转为网络输入尺寸(224)下的像素点坐标
        points_pixel = pred_center * 224.0
        warped_points = warp_points(points_pixel, Minv_pred, device=device, sz=224)

    original_img = postprocess_image(img)
    warped_img = postprocess_image(img_warped)
    
    # 绘制预测的圆心点(红点)
    cx, cy = map(int, points_pixel.squeeze(0).cpu().numpy())
    cv2.circle(original_img, (cx, cy), 3, (0, 0, 255), -1)
    
    # 绘制变换后的圆心点(红点)
    wcx, wcy = map(int, warped_points.squeeze(0).cpu().numpy())
    cv2.circle(warped_img, (wcx, wcy), 3, (0, 0, 255), -1)

    homography_matrix = Minv_pred.squeeze(0).cpu().numpy()

    return original_img, warped_img, homography_matrix


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_stn = load_stn_model(args.model_path, device)
    original_img, warped_img, homography = rectify_clock_image(model_stn, args.input_image, device)

    print("单应矩阵:")
    print(homography)

    combined = np.hstack([original_img, warped_img])
    cv2.imshow("Original (left) vs Warped (right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-i", "--input_image", type=str, required=True)
    args = parser.parse_args()
    main(args)
