import os
import cv2
import numpy as np
from util.augmentation import BaseTransform
from util.config import config as cfg, print_config
from network.textnet import TextNet
from util.detection_mask import TextDetector as TextDetector_mask
import torch
from util.misc import to_device
from util.read_meter import MeterReader
from util.converter import StringLabelConverter
from get_meter_area import Detector
from dataset.stn_transform import STNTransformer

# parse arguments
print_config(cfg)

# Initialize STN if enabled
stn_transformer = None
if cfg.get("use_stn", False):
    stn_model_path = cfg.get("stn_model_path", "")
    print(f"Initializing STN from {stn_model_path}")
    stn_transformer = STNTransformer(stn_model_path, device=cfg.device)

predict_dir = "datas/demo"

model = TextNet(is_training=False, backbone=cfg.net)
model_path = cfg.model_path
model.load_model(model_path)
model = model.to(cfg.device)
converter = StringLabelConverter()

det = Detector()
detector = TextDetector_mask(model)
meter = MeterReader()
transform = BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)


image_list = os.listdir(predict_dir)


for index in image_list:
    print("**************", index)
    image_path = os.path.join(predict_dir, index)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Failed to load image from {image_path}. Skipping.")
        continue

    cv2.imshow("det1", image)
    cv2.waitKey(0)

    # detect meter area
    if cfg.get("use_yolo", False):
        _, _, _, meter_list = det.detect(image, index)
    else:
        meter_list = [image]

    # meter_list = [image]

    if len(meter_list) == 0:
        print("no detected meter")
        continue
    else:
        for i in meter_list:
            if stn_transformer is not None:
                i, _ = stn_transformer(i, None)

            cv2.imshow("det", i)
            cv2.waitKey(0)

            image, _ = transform(i)
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).unsqueeze(0)
            image = to_device(image)
            try:
                output = detector.detect1(image)
            except Exception as e:
                print(f"Detection error: {e}")
                continue

            pointer_pred, dail_pred, text_pred, preds, std_points = (
                output["pointer"],
                output["dail"],
                output["text"],
                output["reco"],
                output["std"],
            )

            # decode predicted text
            pred, preds_size = preds
            if pred is not None:
                _, pred = pred.max(2)
                pred = pred.transpose(1, 0).contiguous().view(-1)
                pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
                pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
                # print("preds", pred_transcripts)
            else:
                pred_transcripts = None

            img_show = image[0].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

            result = meter(img_show, pointer_pred, dail_pred, text_pred, pred_transcripts, std_points)
