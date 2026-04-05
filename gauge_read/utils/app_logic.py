import os

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import morphology

from gauge_read.models.textnet import TextNet
from gauge_read.utils.reader import MeterReader, TextDetector, YOLODetector
from gauge_read.utils.augmentation import BaseTransform
from gauge_read.utils.converter import StringLabelConverter
from gauge_read.utils.logger import logger
from gauge_read.utils.stn_transform import STNTransformer
from gauge_read.utils.tools import to_device


class GaugeApp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.system.device
        self.textnet = None
        self.stn = None
        self.detector = None
        self.converter = StringLabelConverter()
        self.meter_reader = MeterReader()
        self.transform = BaseTransform(size=cfg.data.test_size, mean=cfg.model.means, std=cfg.model.stds)
        self.yolo_detector = None

        self.current_image = None
        self.current_std_points = []
        self.current_pointer_line = None
        self.current_start_value = 0.0
        self.current_end_value = 0.0
        self.current_ratio = 0.0
        self.current_center = None
        self.scale_range = 1.6
        self.yolo_weights_path = None

    @staticmethod
    def notify_error(message):
        logger.debug("GaugeApp error notification: %s", message)

    @staticmethod
    def notify_info(message):
        logger.debug("GaugeApp info notification: %s", message)

    def sync_runtime_from(self, other):
        self.textnet = other.textnet
        self.stn = other.stn
        self.detector = other.detector
        self.converter = other.converter
        self.meter_reader = other.meter_reader
        self.transform = other.transform
        self.yolo_detector = other.yolo_detector
        self.yolo_weights_path = other.yolo_weights_path

    def load_models(self, textnet_path, stn_path=None, yolo_path=None):
        textnet_path = textnet_path or self.cfg.predict.get("model_path", "")
        stn_path = stn_path if stn_path is not None else self.cfg.data.get("stn_model_path", "")
        yolo_path = yolo_path or self.cfg.predict.get("yolo_model_path", "")

        logger.info(
            "Loading models for GaugeApp: textnet=%s, stn=%s, yolo=%s, device=%s",
            textnet_path,
            stn_path or "disabled",
            yolo_path or "default",
            self.device,
        )

        if not textnet_path or not os.path.exists(textnet_path):
            self.textnet = None
            self.detector = None
            msg = f"读数模型文件不存在: {textnet_path}"
            logger.error(msg)
            self.notify_error(msg)
            raise FileNotFoundError(msg)

        logger.info("Loading TextNet from %s", textnet_path)
        self.textnet = TextNet(is_training=False, backbone=self.cfg.model.net, cfg=self.cfg)

        try:
            checkpoint = torch.load(textnet_path, map_location=self.device)
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

            model_dict = self.textnet.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.textnet.load_state_dict(model_dict)
            logger.info(
                "TextNet weights loaded: matched_keys=%s, total_checkpoint_keys=%s", len(pretrained_dict), len(state_dict)
            )
        except Exception as exc:
            self.textnet = None
            self.detector = None
            msg = f"无法加载读数模型: {str(exc)}"
            logger.exception("Failed to load TextNet weights from %s", textnet_path)
            self.notify_error(msg)
            raise

        self.textnet = self.textnet.to(self.device)
        self.textnet.eval()
        self.detector = TextDetector(self.textnet)
        logger.info("TextNet moved to device and switched to eval mode: %s", self.device)

        if stn_path:
            logger.info("Loading STN from %s", stn_path)
            try:
                self.stn = STNTransformer(stn_path, device=self.device)
                if self.stn.model is None:
                    logger.warning("STN initialization completed without an active model: %s", stn_path)
                else:
                    logger.info("STN loaded successfully from %s", stn_path)
            except Exception as exc:
                logger.exception("Failed to initialize STN from %s", stn_path)
                self.notify_error(f"无法加载STN模型: {str(exc)}")
                self.stn = None
        else:
            self.stn = None
            logger.info("STN disabled for GaugeApp")

        try:
            logger.info("Loading YOLO detector from %s", yolo_path if yolo_path else "default")
            self.yolo_detector = YOLODetector(cfg=self.cfg, weights=yolo_path if yolo_path else None)
            self.yolo_weights_path = yolo_path if yolo_path else self.cfg.predict.get("yolo_model_path", "")
            logger.info("YOLO detector initialized successfully: weights=%s", self.yolo_weights_path)
        except Exception as exc:
            logger.exception("Failed to initialize YOLO detector from %s", yolo_path if yolo_path else "default")
            self.notify_error(f"无法加载YOLO模型: {str(exc)}")
            self.yolo_detector = None
            self.yolo_weights_path = None

        self.notify_info(
            f"模型加载完成\n读数模型：{textnet_path}\nSTN模型：{stn_path if stn_path else 'disabled'}\nYOLO模型：{self.yolo_weights_path if self.yolo_detector is not None else 'failed'}"
        )

    def process_image(self, input_image, use_stn=True, use_yolo=False):
        if self.textnet is None:
            logger.error("process_image called before TextNet was loaded")
            return None, "模型未加载", 0.0, 0.0

        logger.info("Starting single-image inference: use_stn=%s, use_yolo=%s", use_stn, use_yolo)

        if isinstance(input_image, Image.Image):
            image = np.array(input_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = input_image.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        logger.debug("Prepared input image for inference with shape=%s", image.shape)

        meter_img = image
        if use_yolo:
            if self.yolo_detector is None:
                logger.error("YOLO inference requested but YOLO detector is not loaded")
                return None, "YOLO模型未正确加载，请先选择YOLO模型并点击加载模型", 0.0, 0.0
            _, _, _, meter_list = self.yolo_detector.detect(image, "web_upload")
            logger.debug("YOLO detection completed: detected_meters=%s", len(meter_list))
            if len(meter_list) > 0:
                meter_img = meter_list[0]
            else:
                logger.warning("YOLO did not detect any meter in the current image")
                return None, "YOLO未检测到仪表", 0.0, 0.0

        processed_img = meter_img
        predicted_center = None
        if self.stn:
            if use_stn:
                stn_img, _, warped_center = self.stn(meter_img, None)
                processed_img = stn_img
                predicted_center = warped_center
                logger.debug("STN correction applied, predicted_center=%s", predicted_center)
            else:
                _, center_pixel = self.stn.get_homography_matrix(meter_img)
                predicted_center = center_pixel
                logger.debug("STN center predicted without warping, center=%s", predicted_center)

        trans_img_np, _ = self.transform(processed_img)

        if predicted_center is not None:
            ori_h, ori_w = processed_img.shape[:2]
            new_h, new_w = trans_img_np.shape[:2]
            cx = predicted_center[0] * (new_w / ori_w)
            cy = predicted_center[1] * (new_h / ori_h)
            predicted_center = (cx, cy)

        img_show = trans_img_np.copy()
        img_show = ((img_show * np.array(self.cfg.model.stds) + np.array(self.cfg.model.means)) * 255).astype(np.uint8)
        display_img = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        trans_img = trans_img_np.transpose(2, 0, 1)
        trans_img = torch.from_numpy(trans_img).unsqueeze(0)
        trans_img = to_device(trans_img, device=self.device)
        logger.debug("Transformed image tensor ready for detector on device=%s", self.device)

        try:
            output = self.detector.detect1(trans_img)
        except Exception as exc:
            logger.exception("Text detector inference failed")
            return None, f"推理错误: {exc}", 0.0, 0.0

        pointer_pred = output["pointer"]
        preds = output["reco"]
        std_points = output["std"]

        pred, preds_size = preds
        pred_transcripts = ""
        if pred is not None:
            _, pred = pred.max(2)
            pred = pred.transpose(1, 0).contiguous().view(-1)
            transcript = self.converter.decode(pred.data, preds_size.data, raw=False)
            pred_transcripts = transcript if isinstance(transcript, str) else transcript[0]

        pointer_line = self._get_pointer_line(pointer_pred, trans_img_np.shape, predicted_center)
        logger.debug("Model returned std_points=%s", std_points)

        final_std = std_points[:2] if std_points and len(std_points) >= 2 else [(0, 0), (0, 0)]

        self.current_image = display_img
        self.current_pointer_line = pointer_line
        self.current_std_points = final_std
        self.current_center = predicted_center
        self.current_start_value = 0.0
        ocr_error = False

        try:
            if isinstance(pred_transcripts, str):
                clean_text = "".join(filter(lambda x: x.isdigit() or x == ".", pred_transcripts))
                if clean_text:
                    self.current_end_value = float(clean_text)
                else:
                    ocr_error = True
                    self.current_end_value = 0.0
            else:
                ocr_error = True
                self.current_end_value = 0.0
        except ValueError:
            self.current_end_value = 0.0
            ocr_error = True
            logger.warning("OCR transcript could not be converted to numeric value: %s", pred_transcripts)

        value = self.recalculate(ocr_error)
        logger.info(
            "Single-image inference completed: value=%s, ratio=%s, start_value=%s, end_value=%s, ocr_error=%s",
            value,
            self.current_ratio,
            self.current_start_value,
            self.current_end_value,
            ocr_error,
        )

        return self.draw_visualization(), value, self.current_start_value, self.current_end_value

    def _get_pointer_line(self, mask, shape, center=None):
        pointer_skeleton = morphology.skeletonize(mask > 0)
        pointer_edges = pointer_skeleton * 255
        pointer_edges = pointer_edges.astype(np.uint8)

        pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=400)

        if pointer_lines is not None and len(pointer_lines) > 0:
            x1, y1, x2, y2 = pointer_lines[0][0]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
        else:
            return [(0, 0), (10, 10)]

        if center is not None:
            cx, cy = center[0], center[1]
            d1 = (pt1[0] - cx) ** 2 + (pt1[1] - cy) ** 2
            d2 = (pt2[0] - cx) ** 2 + (pt2[1] - cy) ** 2
            if d1 > d2:
                return [pt2, pt1]
            return [pt1, pt2]

        return [pt1, pt2]

    def recalculate(self, ocr_error=False):
        if self.textnet is None or self.detector is None:
            return "模型未加载"
        if self.current_image is None:
            return "请先运行推理"
        if not self.current_std_points or len(self.current_std_points) < 2:
            return "缺少标定点"
        if not self.current_pointer_line:
            return "缺少指针"

        value, ratio = self.meter_reader.compute_reading(
            self.current_std_points,
            self.current_pointer_line,
            self.current_start_value,
            self.current_end_value,
            self.current_center,
        )
        self.current_ratio = ratio
        if ocr_error:
            return "OCR error, fallback to ratio-based reading: {:.2f}".format(value)
        logger.info("Calculation completed: value=%s, ratio=%s", value, ratio)
        return value

    def draw_visualization(self):
        if self.current_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        img = self.current_image.copy()

        if len(self.current_std_points) >= 1:
            cv2.circle(img, (int(self.current_std_points[0][0]), int(self.current_std_points[0][1])), 5, (0, 255, 0), -1)
            cv2.putText(
                img,
                "Start",
                (int(self.current_std_points[0][0]) + 10, int(self.current_std_points[0][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if len(self.current_std_points) >= 2:
            cv2.circle(img, (int(self.current_std_points[1][0]), int(self.current_std_points[1][1])), 5, (0, 0, 255), -1)
            cv2.putText(
                img,
                "End",
                (int(self.current_std_points[1][0]) + 10, int(self.current_std_points[1][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        if self.current_pointer_line:
            p1 = tuple(map(int, self.current_pointer_line[0]))
            p2 = tuple(map(int, self.current_pointer_line[1]))
            cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2, tipLength=0.1)

        if self.current_center is not None:
            cx, cy = int(self.current_center[0]), int(self.current_center[1])
            cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(img, "Center", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return img
