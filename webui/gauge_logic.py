import sys
import os
import torch
import cv2
import numpy as np
from PIL import Image

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.config import config as cfg
from network.textnet import TextNet
from util.detection_mask import TextDetector as TextDetector_mask
from util.read_meter import MeterReader
from util.converter import StringLabelConverter
from util.augmentation import BaseTransform
from util.misc import to_device
from dataset.stn_transform import STNTransformer
from get_meter_area import Detector as YoloDetector


class GaugeAppModel:
    def __init__(self):
        self.device = cfg.device
        self.textnet = None
        self.stn = None
        self.detector = None
        self.converter = StringLabelConverter()
        self.meter_reader = MeterReader()
        self.transform = BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
        self.yolo_detector = YoloDetector()

        # Current state
        self.current_image = None
        self.current_std_points = []  # [point1, point2]
        self.current_pointer_line = None  # [start, end]
        self.current_start_value = 0.0  # user input start val
        self.current_end_value = 0.0  # ocr result or user input end val
        self.current_ratio = 0.0
        self.scale_range = 1.6  # Default, maybe user should input this? Or inferred.

    def load_models(self, textnet_path, stn_path=None, yolo_path=None):
        # Load TextNet
        print(f"Loading TextNet from {textnet_path}")
        self.textnet = TextNet(is_training=False, backbone=cfg.net)

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
        except Exception as e:
            return f"Error loading TextNet: {str(e)}"

        self.textnet = self.textnet.to(self.device)
        self.textnet.eval()
        self.detector = TextDetector_mask(self.textnet)

        # Load STN
        if stn_path:
            print(f"Loading STN from {stn_path}")
            try:
                self.stn = STNTransformer(stn_path, device=self.device)
            except Exception as e:
                return f"Error loading STN: {str(e)}"
        else:
            self.stn = None

        # Load YOLO
        try:
            print(f"Loading YOLO Detector from {yolo_path if yolo_path else 'default'}...")
            self.yolo_detector = YoloDetector(weights=yolo_path)
        except Exception as e:
            print(f"Warning: Failed to load YOLO: {e}")
            self.yolo_detector = None

        return "Models loaded successfully!"

    def process_image(self, input_image, use_stn=True, use_yolo=False):
        if input_image is None or self.textnet is None:
            return None, "Model not loaded or no image.", 0.0, 0.0

        # Convert to numpy/cv2 if PIL
        if isinstance(input_image, Image.Image):
            image = np.array(input_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Internal pipeline uses BGR
        else:
            image = input_image.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Gradio usually gives RGB, cv2 needs BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Meter Detection (YOLO)
        meter_img = image
        if use_yolo:
            if self.yolo_detector is None:
                return None, "YOLO selected but not loaded correctly.", 0.0, 0.0
            # Assuming 'index' param in detector.detect is just for debug print/mask saving
            # We pass "web_upload"
            _, _, _, meter_list = self.yolo_detector.detect(image, "web_upload")
            if len(meter_list) > 0:
                meter_img = meter_list[0]  # Take first detected meter
            else:
                return None, "YOLO enabled but no meter detected.", 0.0, 0.0

        # 2. STN
        processed_img = meter_img
        if use_stn and self.stn:
            processed_img, _ = self.stn(meter_img, None)

        # 3. TextNet Inference
        trans_img, _ = self.transform(processed_img)
        trans_img = trans_img.transpose(2, 0, 1)
        trans_img = torch.from_numpy(trans_img).unsqueeze(0)
        trans_img = to_device(trans_img)

        try:
            output = self.detector.detect1(trans_img)
        except Exception as e:
            return None, f"Inference Error: {e}", 0.0, 0.0

        # Unpack
        # Note: detect1 logic was modified in previous turn to return aux map?
        # If user undid changes, `detect1` returns old tuple.
        # We need to handle both cases safely.
        res = output
        if isinstance(res, dict):
            pointer_pred = res["pointer"]
            dail_pred = res["dail"]
            text_pred = res["text"]
            preds = res["reco"]
            std_points = res["std"]  # This is usually returning None/Empty in raw `detect1`?
            # Wait, `detect1` calls `forward_test`.
            # In `textnet.py`, `forward_test` returns `std_points` calculated from contours.
            # So `std_points` here should be a list of tuples [(x,y), (x,y)]

        # Decode Text
        pred, preds_size = preds
        pred_transcripts = ""
        if pred is not None:
            _, pred = pred.max(2)
            pred = pred.transpose(1, 0).contiguous().view(-1)
            t = self.converter.decode(pred.data, preds_size.data, raw=False)
            pred_transcripts = t if isinstance(t, str) else t[0]

        # Prepare for geometric calculation
        # We need to run the `MeterReader` logic to get the pointer line and final value
        # But `MeterReader` takes masks and does skeletonization.

        # Prepare display image (RGB for Gradio)
        display_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        # Use MeterReader to get lines and value
        # We need to invoke `meter_reader.__call__` logic but extract intermediate data
        # Let's peek into `read_meter.py` again. `find_lines` returns `value`.
        # It doesn't easily return the pointer coordinates.
        # We should modify `read_meter.py` or duplicate logic here to get the pointer line coordinates.

        # Duplicate/Adapted Logic for control:
        p_mask = pointer_pred
        d_mask = dail_pred

        # 1. Get Pointer Line
        pointer_line = self._get_pointer_line(p_mask, processed_img.shape)

        # 2. Get Std Points if not valid from model
        # The model tries to find them. If `std_points` from model is valid (len>=2), use it.
        # Note: `std_points` in `forward_test` output might be [std_point[0], ref_point[0]] or similar
        print(f"Model returned std_points: {std_points}")

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
        # Note: current logic assumes P1 is always 0.
        self.current_start_value = 0.0

        # Calculate initial reading
        try:
            # Try to convert OCR text to number
            if isinstance(pred_transcripts, str):
                clean_text = "".join(filter(lambda x: x.isdigit() or x == ".", pred_transcripts))
                self.current_end_value = float(clean_text) if clean_text else 0.0
        except:
            self.current_end_value = 0.0

        val = self.recalculate()

        return display_img, val, self.current_start_value, self.current_end_value

    def _get_pointer_line(self, mask, shape):
        # Skeletonize and find line
        from skimage import morphology

        skeleton = morphology.skeletonize(mask > 0)
        # Find endpoints or fit line?
        # Simple method: Probabilistic Hough Line or just fitLine on non-zero points
        y, x = np.where(skeleton)
        if len(x) < 10:
            # Fallback: centroids
            y, x = np.where(mask > 0.5)

        if len(x) < 2:
            return [(0, 0), (10, 10)]

        pts = np.column_stack((x, y))

        # Fit line [vx, vy, x0, y0]
        rows, cols = shape[:2]
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

        # Extrapolate to draw
        # We want a segment that covers the mask
        # Project points onto the line
        # This is simplifying; essentially we want the "Tip" and "Tail"
        # Let's assume the pointer is roughly centered or we find the extremes along the direction

        # Simple approach: Find min and max projection along direction
        vec = np.array([vx, vy]).flatten()
        p0 = np.array([x0, y0]).flatten()

        projections = np.dot(pts - p0, vec)
        min_p = p0 + vec * np.min(projections)
        max_p = p0 + vec * np.max(projections)

        return [(int(min_p[0]), int(min_p[1])), (int(max_p[0]), int(max_p[1]))]

    def recalculate(self):
        """
        Calculate reading based on current state:
        - self.current_std_points (Start Scale, End Scale/Number Point)
        - self.current_start_value (Value at Point 1)
        - self.current_end_value (Value at Point 2)
        - self.current_pointer_line (Tail, Tip)
        """
        if not self.current_std_points or len(self.current_std_points) < 2:
            return "Error: Missing Scale Points"
        if not self.current_pointer_line:
            return "Error: Missing Pointer"

        # Use centralized logic from MeterReader
        val, ratio = self.meter_reader.compute_reading(
            self.current_std_points, self.current_pointer_line, self.current_start_value, self.current_end_value
        )
        self.current_ratio = ratio

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
                f"Start({self.current_start_value})",
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
                f"End({self.current_end_value})",
                (int(self.current_std_points[1][0]) + 10, int(self.current_std_points[1][1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # Draw Pointer
        if self.current_pointer_line:
            p1 = tuple(map(int, self.current_pointer_line[0]))
            p2 = tuple(map(int, self.current_pointer_line[1]))
            cv2.line(img, p1, p2, (255, 0, 0), 2)  # Blue Pointer
            # Center
            cv2.line(img, (int(img.shape[1] / 2), 0), (int(img.shape[1] / 2), img.shape[0]), (100, 100, 100), 1)
            cv2.circle(img, p2, 3, (255, 255, 0), -1)  # Tip

        return img

    def update_point(self, point_type, x, y):
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

        return self.draw_visualization(), self.recalculate()

    def update_start_val(self, text):
        try:
            self.current_start_value = float(text)
        except:
            pass
        return self.draw_visualization(), self.recalculate()

    def update_end_val(self, text):
        try:
            self.current_end_value = float(text)
        except:
            pass
        return self.draw_visualization(), self.recalculate()
