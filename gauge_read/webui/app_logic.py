import os
import torch
import cv2
import numpy as np
import gradio as gr
from PIL import Image
from skimage import morphology

from gauge_read.models.textnet import TextNet
from gauge_read.utils.reader import MeterReader, TextDetector
from gauge_read.utils.converter import StringLabelConverter
from gauge_read.utils.augmentation import BaseTransform
from gauge_read.utils.tools import to_device
from gauge_read.utils.stn_transform import STNTransformer
from gauge_read.utils.logger import logger
from gauge_read.inference import Detector
from gauge_read.webui.batch_infer import BatchInferenceService


class GaugeAppModel:
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

        # Current state
        self.current_image = None
        self.current_std_points = []  # [point1, point2]
        self.current_pointer_line = None  # [start, end]
        self.current_start_value = 0.0  # user input start val
        self.current_end_value = 0.0  # ocr result or user input end val
        self.current_ratio = 0.0
        self.scale_range = 1.6  # Default, maybe user should input this? Or inferred.
        self.yolo_weights_path = None
        self.batch_inference = BatchInferenceService(self)

    def process_batch_directory(self, input_dir, use_stn=True, use_yolo=False, progress=None):
        return self.batch_inference.process_directory(input_dir, use_stn=use_stn, use_yolo=use_yolo, progress=progress)

    def load_models(self, textnet_path, stn_path=None, yolo_path=None):
        textnet_path = textnet_path or self.cfg.predict.get("model_path", "")
        stn_path = stn_path if stn_path is not None else self.cfg.data.get("stn_model_path", "")
        yolo_path = yolo_path or self.cfg.predict.get("yolo_model_path", "")

        logger.info(
            "Loading models for GaugeAppModel: textnet=%s, stn=%s, yolo=%s, device=%s",
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
            gr.Error(msg)
            raise FileNotFoundError(msg)

        # Load TextNet
        logger.info("Loading TextNet from %s", textnet_path)
        self.textnet = TextNet(is_training=False, backbone=self.cfg.model.net, cfg=self.cfg)

        # Robust loading context
        try:
            checkpoint = torch.load(textnet_path, map_location=self.device)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Filter logic similar to what we tried to patch
            model_dict = self.textnet.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            self.textnet.load_state_dict(model_dict)
            logger.info(
                "TextNet weights loaded: matched_keys=%s, total_checkpoint_keys=%s", len(pretrained_dict), len(state_dict)
            )
        except Exception as e:
            self.textnet = None
            self.detector = None
            msg = f"无法加载读数模型: {str(e)}"
            logger.exception("Failed to load TextNet weights from %s", textnet_path)
            gr.Error(msg)
            raise

        self.textnet = self.textnet.to(self.device)
        self.textnet.eval()
        self.detector = TextDetector(self.textnet)
        logger.info("TextNet moved to device and switched to eval mode: %s", self.device)

        # Load STN
        if stn_path:
            logger.info("Loading STN from %s", stn_path)
            try:
                self.stn = STNTransformer(stn_path, device=self.device)
                if self.stn.model is None:
                    logger.warning("STN initialization completed without an active model: %s", stn_path)
                else:
                    logger.info("STN loaded successfully from %s", stn_path)
            except Exception as e:
                logger.exception("Failed to initialize STN from %s", stn_path)
                gr.Error(f"无法加载STN模型: {str(e)}")
                self.stn = None
        else:
            self.stn = None
            logger.info("STN disabled for GaugeAppModel")

        # Load YOLO
        try:
            logger.info("Loading YOLO detector from %s", yolo_path if yolo_path else "default")
            self.yolo_detector = Detector(cfg=self.cfg, weights=yolo_path if yolo_path else None)
            self.yolo_weights_path = yolo_path if yolo_path else self.cfg.predict.get("yolo_model_path", "")
            logger.info("YOLO detector initialized successfully: weights=%s", self.yolo_weights_path)
        except Exception as e:
            logger.exception("Failed to initialize YOLO detector from %s", yolo_path if yolo_path else "default")
            gr.Error(f"无法加载YOLO模型: {str(e)}")
            self.yolo_detector = None
            self.yolo_weights_path = None

        gr.Info(
            f"模型加载完成\n读数模型：{textnet_path}\nSTN模型：{stn_path if stn_path else 'disabled'}\nYOLO模型：{self.yolo_weights_path if self.yolo_detector is not None else 'failed'}"
        )

    def process_image(self, input_image, use_stn=True, use_yolo=False):
        if self.textnet is None:
            logger.error("process_image called before TextNet was loaded")
            return None, "模型未加载", 0.0, 0.0

        logger.info("Starting single-image inference: use_stn=%s, use_yolo=%s", use_stn, use_yolo)

        # Convert to numpy/cv2 if PIL
        if isinstance(input_image, Image.Image):
            image = np.array(input_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Internal pipeline uses BGR
        else:
            image = input_image.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Gradio usually gives RGB, cv2 needs BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        logger.debug("Prepared input image for inference with shape=%s", image.shape)

        # 1. Meter Detection (YOLO)
        meter_img = image
        if use_yolo:
            if self.yolo_detector is None:
                logger.error("YOLO inference requested but YOLO detector is not loaded")
                return None, "YOLO模型未正确加载，请先选择YOLO模型并点击加载模型", 0.0, 0.0
            # Assuming 'index' param in detector.detect is just for debug print/mask saving
            # We pass "web_upload"
            _, _, _, meter_list = self.yolo_detector.detect(image, "web_upload")
            logger.debug("YOLO detection completed: detected_meters=%s", len(meter_list))
            if len(meter_list) > 0:
                meter_img = meter_list[0]  # Take first detected meter
            else:
                logger.warning("YOLO did not detect any meter in the current image")
                return None, "YOLO未检测到仪表", 0.0, 0.0

        # 2. STN
        processed_img = meter_img
        predicted_center = None

        # 无论是否使用STN变形，只要有STN模型，都可以用来预测圆心
        if self.stn:
            if use_stn:
                stn_img, _, warped_center = self.stn(meter_img, None)
                processed_img = stn_img
                predicted_center = warped_center
                logger.debug("STN correction applied, predicted_center=%s", predicted_center)
            else:
                # 不使用变形时，依然获取在原图(未变形)坐标系下的圆心
                _, center_pixel = self.stn.get_homography_matrix(meter_img)
                predicted_center = center_pixel
                logger.debug("STN center predicted without warping, center=%s", predicted_center)

        # 3. TextNet Inference (使用 BaseTransform 获取统一坐标系和规范化的输入)
        trans_img_np, _ = self.transform(processed_img)

        # 核心：将预测的圆心坐标同步映射缩放到 TextNet 推理的特征图尺寸上
        if predicted_center is not None:
            ori_h, ori_w = processed_img.shape[:2]
            new_h, new_w = trans_img_np.shape[:2]
            cx = predicted_center[0] * (new_w / ori_w)
            cy = predicted_center[1] * (new_h / ori_h)
            predicted_center = (cx, cy)

        # 构建用于界面展示和画图的 display_img，反向归一化 trans_img_np
        img_show = trans_img_np.copy()
        img_show = ((img_show * np.array(self.cfg.model.stds) + np.array(self.cfg.model.means)) * 255).astype(np.uint8)
        display_img = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)

        trans_img = trans_img_np.transpose(2, 0, 1)
        trans_img = torch.from_numpy(trans_img).unsqueeze(0)
        trans_img = to_device(trans_img, device=self.device)
        logger.debug("Transformed image tensor ready for detector on device=%s", self.device)

        try:
            output = self.detector.detect1(trans_img)
        except Exception as e:
            logger.exception("Text detector inference failed")
            return None, f"推理错误: {e}", 0.0, 0.0

        res = output
        if isinstance(res, dict):
            pointer_pred = res["pointer"]
            preds = res["reco"]
            std_points = res["std"]

        # Decode Text
        pred, preds_size = preds
        pred_transcripts = ""
        if pred is not None:
            _, pred = pred.max(2)
            pred = pred.transpose(1, 0).contiguous().view(-1)
            t = self.converter.decode(pred.data, preds_size.data, raw=False)
            pred_transcripts = t if isinstance(t, str) else t[0]

        # Duplicate/Adapted Logic for control:
        p_mask = pointer_pred

        # 1. Get Pointer Line
        pointer_line = self._get_pointer_line(p_mask, trans_img_np.shape, predicted_center)

        # 2. Get Std Points if not valid from model
        # The model tries to find them. If `std_points` from model is valid (len>=2), use it.
        # Note: `std_points` in `forward_inference` output might be [std_point[0], ref_point[0]] or similar
        logger.debug("Model returned std_points=%s", std_points)

        final_std = []
        if std_points and len(std_points) >= 2:
            final_std = std_points[:2]
        else:
            # Default fallback?
            final_std = [(0, 0), (0, 0)]

        # Store state
        self.current_image = display_img
        self.current_pointer_line = pointer_line  # [start(x,y), end(x,y)]
        self.current_std_points = final_std
        self.current_center = predicted_center
        # Note: current logic assumes P1 is always 0.
        self.current_start_value = 0.0
        ocr_error = False

        # Calculate initial reading
        try:
            # Try to convert OCR text to number
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

        val = self.recalculate(ocr_error)
        logger.info(
            "Single-image inference completed: value=%s, ratio=%s, start_value=%s, end_value=%s, ocr_error=%s",
            val,
            self.current_ratio,
            self.current_start_value,
            self.current_end_value,
            ocr_error,
        )

        return self.draw_visualization(), val, self.current_start_value, self.current_end_value

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
                # pt2 is closer to center -> pt2 is root, pt1 is tip
                return [pt2, pt1]
            else:
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

        # Use centralized logic from MeterReader
        val, ratio = self.meter_reader.compute_reading(
            self.current_std_points,
            self.current_pointer_line,
            self.current_start_value,
            self.current_end_value,
            getattr(self, "current_center", None),
        )
        self.current_ratio = ratio
        if ocr_error:
            return "OCR error, fallback to ratio-based reading: {:.2f}".format(val)
        logger.info("Calculation completed: value=%s, ratio=%s", val, ratio)
        return val

    def draw_visualization(self):
        if self.current_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        img = self.current_image.copy()

        # Draw Std Points
        if len(self.current_std_points) >= 1:
            cv2.circle(
                img, (int(self.current_std_points[0][0]), int(self.current_std_points[0][1])), 5, (0, 255, 0), -1
            )  # Green Start
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
            cv2.circle(
                img, (int(self.current_std_points[1][0]), int(self.current_std_points[1][1])), 5, (0, 0, 255), -1
            )  # Red End
            cv2.putText(
                img,
                "End",
                (int(self.current_std_points[1][0]) + 10, int(self.current_std_points[1][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # Draw Pointer
        if self.current_pointer_line:
            p1 = tuple(map(int, self.current_pointer_line[0]))  # root
            p2 = tuple(map(int, self.current_pointer_line[1]))  # tip

            # 使用带有箭头的线段代替原本的普通直线和圆点
            cv2.arrowedLine(img, p1, p2, (255, 0, 0), 2, tipLength=0.1)  # Blue Pointer with Arrow

        # Draw Center
        if getattr(self, "current_center", None) is not None:
            cx, cy = int(self.current_center[0]), int(self.current_center[1])
            cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)  # Yellow Center
            cv2.putText(img, "Center", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return img

    def update_point(self, point_type, x, y):
        if self.textnet is None or self.detector is None:
            return gr.skip(), "模型未加载"
        if self.current_image is None:
            return gr.skip(), "请先运行推理"

        x, y = int(x), int(y)
        if point_type == "start":
            if not self.current_std_points:
                self.current_std_points = [(0, 0), (0, 0)]
            self.current_std_points[0] = (x, y)
        elif point_type == "end":
            if len(self.current_std_points) < 2:
                self.current_std_points.append((0, 0))
            self.current_std_points[1] = (x, y)
        elif point_type == "pointer_tip":
            if not self.current_pointer_line:
                self.current_pointer_line = [(0, 0), (0, 0)]
            self.current_pointer_line[1] = (x, y)
        elif point_type == "pointer_root":
            if not self.current_pointer_line:
                self.current_pointer_line = [(0, 0), (0, 0)]
            self.current_pointer_line[0] = (x, y)
        elif point_type == "center":
            self.current_center = (x, y)

        return self.draw_visualization(), self.recalculate()

    def update_start_val(self, text):
        if self.textnet is None or self.detector is None:
            return gr.skip(), "模型未加载"
        if self.current_image is None:
            return gr.skip(), "请先运行推理"

        try:
            self.current_start_value = float(text)
        except ValueError:
            return gr.skip(), "起始值输入无效"
        return self.draw_visualization(), self.recalculate()

    def update_end_val(self, text):
        if self.textnet is None or self.detector is None:
            return gr.skip(), "模型未加载"
        if self.current_image is None:
            return gr.skip(), "请先运行推理"

        try:
            self.current_end_value = float(text)
        except ValueError:
            return gr.skip(), "结束值输入无效"
        return self.draw_visualization(), self.recalculate()
