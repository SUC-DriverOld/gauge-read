import os
from random import randint

import cv2
import numpy as np
import torch
from yolov5.models.experimental import attempt_load
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

from gauge_read.models.textnet import TextNet
from gauge_read.utils.stn_transform import STNTransformer
from gauge_read.utils.augmentation import BaseTransform
from gauge_read.utils.config import config as cfg, print_config
from gauge_read.utils.converter import StringLabelConverter
from gauge_read.utils.tools import to_device
from gauge_read.utils.reader import MeterReader, TextDetector


class Detector(object):
    def __init__(self, weights=None):
        self.img_size = 640
        self.threshold = 0.6
        self.max_frame = 160
        self.weights = weights if weights else "pretrain/best.pt"
        self.init_model()

    def init_model(self):
        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        self.m = model
        self.names = model.module.names if hasattr(model, "module") else model.names
        self.colors = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in self.names]

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img0, img

    def plot_bboxes(self, image, bboxes, line_thickness=None):
        tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        for x1, y1, x2, y2, cls_id, conf in bboxes:
            color = self.colors[self.names.index(cls_id)]
            c1, c2 = (x1, y1), (x2, y2)
            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                "{} ID-{:.2f}".format(cls_id, conf),
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        return image

    def detect(self, im, _):
        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.3)

        pred_boxes = []
        image_info = {}
        count = 0

        digital_list, meter_list = [], []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]

                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])

                    region = im0[y1:y2, x1:x2]
                    if lbl == "meter":
                        meter_list.append(region)
                    else:
                        digital_list.append(region)

                    pred_boxes.append((x1, y1, x2, y2, lbl, conf))
                    count += 1
                    key = "{}-{:02}".format(lbl, count)
                    image_info[key] = ["{}x{}".format(x2 - x1, y2 - y1), np.round(float(conf), 3)]

        im = self.plot_bboxes(im, pred_boxes)
        return im, image_info, digital_list, meter_list


def main():
    print_config(cfg)

    stn_transformer = None
    if cfg.predict.get("use_stn", False):
        stn_model_path = cfg.data.get("stn_model_path", "")
        print(f"Initializing STN from {stn_model_path}")
        stn_transformer = STNTransformer(stn_model_path, device=cfg.system.device)

    predict_dir = "datas/demo"

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
    main()
