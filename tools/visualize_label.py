import numpy as np
import cv2
import argparse

from gauge_read.datasets.augmentation import Augmentation
from gauge_read.datasets.meter_data import MeterDataset
from gauge_read.utils.config import AttrDict


def to_black_white(mask):
    mask = np.asarray(mask)
    max_value = np.max(mask)
    if max_value <= 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    return np.array(mask * 255 / max_value, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauge inference")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to YAML config file. If omitted, default config is used."
    )
    args = parser.parse_args()
    cfg = AttrDict(args.config or AttrDict.DEFAULT_CONFIG_PATH)
    cfg.print_config()

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(size=640, mean=means, std=stds)
    trainset = MeterDataset(transform=transform, cfg=cfg)

    for idx in range(0, len(trainset)):
        img, pointer_mask, dail_mask, text_mask, train_mask, bboxs, transcripts = trainset[idx]

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        cv2.imshow("imgs", img)
        cv2.imshow("pointer_mask", to_black_white(pointer_mask))
        cv2.imshow("dail_mask", to_black_white(dail_mask))
        cv2.imshow("text_mask", to_black_white(text_mask))

        cv2.waitKey(0)
