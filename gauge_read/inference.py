import argparse
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from gauge_read.models.textnet import TextNet
from gauge_read.utils.stn_transform import STNTransformer
from gauge_read.utils.augmentation import BaseTransform
from gauge_read.utils.config import config as cfg, load_config, print_config
from gauge_read.utils.converter import StringLabelConverter
from gauge_read.utils.tools import to_device
from gauge_read.utils.reader import MeterReader, TextDetector


class Detector(object):
    def __init__(self, weights=None):
        yolo_cfg = cfg.get("predict", {})
        self.img_size = int(yolo_cfg.get("yolo_imgsz", 640))
        self.threshold = float(yolo_cfg.get("yolo_conf", 0.6))
        self.iou_threshold = float(yolo_cfg.get("yolo_iou", 0.3))
        self.max_det = int(yolo_cfg.get("yolo_max_det", 160))
        self.device = yolo_cfg.get("yolo_device", "auto")
        self.half = yolo_cfg.get("yolo_half", None)
        self.weights = weights if weights else yolo_cfg.get("yolo_model_path", "pretrain/best.pt")
        self.init_model()

    def init_model(self):
        if self.device in (None, "", "auto"):
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if self.half is None:
            # Default to FP16 on CUDA and FP32 on CPU.
            self.use_half = str(self.device).startswith("cuda")
        else:
            self.use_half = bool(self.half)

        self.m = YOLO(self.weights)
        self.names = self.m.names

    def _get_label(self, cls_id):
        if isinstance(self.names, dict):
            return self.names.get(cls_id, str(cls_id))
        if isinstance(self.names, (list, tuple)) and 0 <= cls_id < len(self.names):
            return self.names[cls_id]
        return str(cls_id)

    def detect(self, im, _):
        im0 = im.copy()
        results = self.m.predict(
            source=im0,
            imgsz=self.img_size,
            conf=self.threshold,
            iou=self.iou_threshold,
            device=self.device,
            half=self.use_half,
            max_det=self.max_det,
            verbose=False,
        )

        image_info = {}
        count = 0
        digital_list, meter_list = [], []

        if not results:
            return im0, image_info, digital_list, meter_list

        result = results[0]
        boxes = result.boxes
        annotated = result.plot()

        if boxes is None or len(boxes) == 0:
            return annotated, image_info, digital_list, meter_list

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.int32)
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(np.int32)
        h, w = im0.shape[:2]

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            cls_id = int(classes[i])
            conf = float(confs[i])
            lbl = self._get_label(cls_id)

            # Clip coordinates to image bounds to avoid invalid slicing.
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            region = im0[y1:y2, x1:x2]
            if lbl == "pointer":
                meter_list.append(region)
            else:
                digital_list.append(region)

            count += 1
            key = "{}-{:02}".format(lbl, count)
            image_info[key] = ["{}x{}".format(x2 - x1, y2 - y1), np.round(conf, 3)]

        return annotated, image_info, digital_list, meter_list


def main(args):
    if args.config:
        load_config(args.config)

    print_config(cfg)

    stn_transformer = None
    if cfg.predict.get("use_stn", False):
        stn_model_path = cfg.data.get("stn_model_path", "")
        print(f"Initializing STN from {stn_model_path}")
        stn_transformer = STNTransformer(stn_model_path, device=cfg.system.device)

    predict_dir = args.data_dir or cfg.predict.get("data_dir", "datas/demo")

    if not os.path.isdir(predict_dir):
        raise NotADirectoryError(f"Predict directory not found: {predict_dir}")

    model = TextNet(is_training=False, backbone=cfg.model.net)
    model_path = cfg.predict.model_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Please update `predict.model_path` in gauge_read/configs/config.yaml or provide a valid config file."
        )
    model.load_model(model_path)
    model = model.to(cfg.system.device)
    converter = StringLabelConverter()

    det = Detector()
    detector = TextDetector(model)
    meter = MeterReader()
    transform = BaseTransform(size=cfg.data.test_size, mean=cfg.model.means, std=cfg.model.stds)

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

        if cfg.predict.get("use_yolo", False):
            _, _, _, meter_list = det.detect(image, index)
        else:
            meter_list = [image]

        if len(meter_list) == 0:
            print("no detected meter")
            continue

        for meter_img in meter_list:
            predicted_center = None
            if stn_transformer is not None:
                meter_img, _, predicted_center = stn_transformer(meter_img, None)

            cv2.imshow("det", meter_img)
            cv2.waitKey(0)

            norm_img, _ = transform(meter_img)
            norm_img = norm_img.transpose(2, 0, 1)
            norm_img = torch.from_numpy(norm_img).unsqueeze(0)
            norm_img = to_device(norm_img)
            try:
                output = detector.detect1(norm_img)
            except Exception as e:
                print(f"Detection error: {e}")
                continue

            pointer_pred, dail_pred, text_pred, preds, std_points, aux_map = (
                output["pointer"],
                output["dail"],
                output["text"],
                output["reco"],
                output["std"],
                output["aux"],
            )

            if aux_map is not None:
                cv2.imshow("aux_blackhat", aux_map)

            pred, preds_size = preds
            if pred is not None:
                _, pred = pred.max(2)
                pred = pred.transpose(1, 0).contiguous().view(-1)
                pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
                pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
            else:
                pred_transcripts = None

            img_show = norm_img[0].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * np.array(cfg.model.stds) + np.array(cfg.model.means)) * 255).astype(np.uint8)

            meter(
                img_show,
                pointer_pred,
                dail_pred,
                text_pred,
                pred_transcripts,
                std_points,
                getattr(stn_transformer, "last_center", predicted_center),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauge inference")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If omitted, default config is used.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing images to predict. Overrides config/default path.",
    )
    args = parser.parse_args()
    main(args)
