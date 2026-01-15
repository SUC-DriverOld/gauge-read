import numpy as np
import cv2
from util.augmentation import Augmentation
import time
from dataset.meter_data import Meter
import matplotlib.pyplot as plt


def heatmap(im_gray):
    cmap = plt.get_cmap("jet")
    rgba_img = cmap(255 - im_gray)
    Hmap = np.delete(rgba_img, 3, 2)
    return Hmap


if __name__ == "__main__":
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(size=640, mean=means, std=stds)

    trainset = Meter(transform=transform)

    for idx in range(0, len(trainset)):
        t0 = time.time()
        print("idx", idx)

        img, pointer_mask, dail_mask, text_mask, train_mask, bboxs, transcripts = trainset[idx]

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        print("trans", transcripts)
        cv2.imshow("imgs", img)
        cv2.imshow("pointer_mask", heatmap(np.array(pointer_mask * 255 / np.max(pointer_mask), dtype=np.uint8)))
        cv2.imshow("dail_mask", heatmap(np.array(dail_mask * 255 / np.max(dail_mask), dtype=np.uint8)))
        cv2.imshow("text_mask", heatmap(np.array(text_mask * 255 / np.max(text_mask), dtype=np.uint8)))

        cv2.waitKey(0)
