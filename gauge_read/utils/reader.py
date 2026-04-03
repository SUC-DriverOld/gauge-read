import cv2
import numpy as np
import torch
from skimage import morphology

from gauge_read.utils.logger import logger


class TextDetector(object):
    def __init__(self, model):
        self.model = model
        model.eval()
        logger.debug("TextDetector initialized with model=%s", model.__class__.__name__)

    def detect1(self, image):
        logger.debug("TextDetector.detect1 start: image_shape=%s", tuple(image.shape))
        with torch.no_grad():
            pointer_pred, dail_pred, text_pred, pred_recog, std_points, aux_map = self.model.forward_inference(image)

        logger.debug(
            "TextDetector.detect1 finished: pointer_pixels=%s dail_pixels=%s text_pixels=%s std_points=%s has_recog=%s has_aux=%s",
            int(pointer_pred.sum()) if pointer_pred is not None else None,
            int(dail_pred.sum()) if dail_pred is not None else None,
            int(text_pred.sum()) if text_pred is not None else None,
            std_points,
            pred_recog[0] is not None if pred_recog is not None else False,
            aux_map is not None,
        )

        image = image[0].data.cpu().numpy()

        output = {
            "image": image,
            "pointer": pointer_pred,
            "dail": dail_pred,
            "text": text_pred,
            "reco": pred_recog,
            "std": std_points,
            "aux": aux_map,
        }
        return output


class MeterReader(object):
    def __init__(self, debug=False):
        self.debug = debug

    def compute_reading(self, std_points, pointer_line, start_val=0.0, end_val=0.0, predicted_center=None):
        """
        Compute the reading value based on scale points and pointer line.

        Args:
            std_points: List of two points [(x1, y1), (x2, y2)] representing start and end of scale.
            pointer_line: List of two points [(root_x, root_y), (tip_x, tip_y)] representing the pointer.
            start_val: The numeric value at std_points[0].
            end_val: The numeric value at std_points[1].
            predicted_center: Optional tuple/list for meter center (cx, cy).

        Returns:
            float: The calculated reading value.
        """
        if not std_points or len(std_points) < 2:
            logger.warning("compute_reading skipped: invalid std_points=%s", std_points)
            return float(start_val), 0.0
        if not pointer_line or len(pointer_line) < 2:
            logger.warning("compute_reading skipped: invalid pointer_line=%s", pointer_line)
            return float(start_val), 0.0

        p1 = np.array(std_points[0])
        p2 = np.array(std_points[1])
        ptr_root = np.array(pointer_line[0])
        ptr_tip = np.array(pointer_line[1])

        # Calculate Center
        if predicted_center is not None:
            center = np.array(predicted_center)
        else:
            center_coords = self.calculate_center_from_geometry(std_points, pointer_line)
            if center_coords is None:
                logger.debug("compute_reading fallback to pointer root as center")
                center = ptr_root
            else:
                center = np.array(center_coords)

        # Helper for angle (using arctan2 for robustness)
        def get_angle(pt, c):
            return np.arctan2(pt[1] - c[1], pt[0] - c[0]) * 180 / np.pi

        ang_start = get_angle(p1, center)
        ang_end = get_angle(p2, center)
        ang_ptr = get_angle(ptr_tip, center)

        # Helper for positive angle difference
        def diff_angle(target, base):
            d = target - base
            if d < 0:
                d += 360
            return d

        total_angle = diff_angle(ang_end, ang_start)
        ptr_angle = diff_angle(ang_ptr, ang_start)

        if total_angle < 1e-3:
            logger.warning(
                "compute_reading skipped: degenerate total_angle start=%s end=%s center=%s",
                p1.tolist(),
                p2.tolist(),
                center.tolist(),
            )
            return float(start_val), 0.0

        ratio = ptr_angle / total_angle
        val_span = end_val - start_val
        reading = start_val + (ratio * val_span)

        return float(f"{reading:.4f}"), float(f"{ratio:.4f}")

    def __call__(self, image, point_mask, dail_mask, word_mask, number, std_point, predicted_center=None):
        img_result = image.copy()
        value = self.find_lines(img_result, point_mask, dail_mask, word_mask, number, std_point, predicted_center)
        logger.debug("meter reader result=%s", value)

        return value

    def find_lines(self, ori_img, pointer_mask, dail_mask, word_mask, number, std_point, predicted_center=None):
        # 实施骨架算法
        pointer_skeleton = morphology.skeletonize(pointer_mask > 0)
        pointer_edges = pointer_skeleton * 255
        pointer_edges = pointer_edges.astype(np.uint8)
        # cv2.imshow("pointer_edges", pointer_edges)
        # cv2.waitKey(0)

        dail_mask = np.clip(dail_mask, 0, 1)
        dail_edges = dail_mask * 255
        dail_edges = dail_edges.astype(np.uint8)
        # cv2.imshow("dail_edges", dail_edges)
        # cv2.waitKey(0)

        if self.debug:
            # Draw masks for visualization (Red for pointer, Green for dial)
            colored_masks = np.zeros_like(ori_img)
            colored_masks[pointer_mask > 0] = [0, 0, 255]  # Red for pointer
            colored_masks[dail_mask > 0] = [0, 255, 0]  # Green for dial
            colored_masks[word_mask > 0] = [255, 0, 0]  # Blue for word
            ori_img = cv2.addWeighted(ori_img, 1.0, colored_masks, 0.5, 0)

        pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=400)
        if pointer_lines is None or len(pointer_lines) == 0:
            logger.warning("pointer detection failed: no Hough line candidates")
            return "can not detect pointer"

        # Pick the longest candidate for better robustness than taking the first line.
        lines = pointer_lines[:, 0, :]
        lengths = (lines[:, 2] - lines[:, 0]) ** 2 + (lines[:, 3] - lines[:, 1]) ** 2
        best_idx = int(np.argmax(lengths))
        x1, y1, x2, y2 = lines[best_idx]
        coin1 = (x1, y1)
        coin2 = (x2, y2)
        if self.debug:
            cv2.line(ori_img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        # Default center (image center)
        h, w, _ = ori_img.shape
        center = (0.5 * w, 0.5 * h)

        if predicted_center is not None:
            center = (int(predicted_center[0]), int(predicted_center[1]))
            # Draw predicted center
            if self.debug:
                cv2.circle(ori_img, center, 5, (0, 0, 255), -1)
            logger.debug("using predicted center from STN: %s", center)
        else:
            # Try to calculate a better center using geometry:
            # Intersection of pointer line and the perpendicular bisector of the two scale points
            if std_point is not None:
                geo_center = self.calculate_center_from_geometry(std_point, (coin1, coin2))
                if geo_center is not None:
                    # Sanity check: Center should be reasonably close to image area
                    if -w < geo_center[0] < 2 * w and -h < geo_center[1] < 2 * h:
                        center = geo_center
                        # Draw calculated center
                        if self.debug:
                            cv2.circle(ori_img, center, 5, (0, 255, 255), -1)

        dis1 = (coin1[0] - center[0]) ** 2 + (coin1[1] - center[1]) ** 2
        dis2 = (coin2[0] - center[0]) ** 2 + (coin2[1] - center[1]) ** 2
        if dis1 <= dis2:
            pointer_line = (coin1, coin2)
        else:
            pointer_line = (coin2, coin1)

        # print("pointer_line", pointer_line)

        if std_point is None:
            logger.warning("dial detection failed: std_point is missing")
            return "can not detect dail"

        # calculate angle
        a1 = std_point[0]
        a2 = std_point[1]
        if self.debug:
            cv2.circle(ori_img, a1, 2, (255, 0, 0), 2)
            cv2.circle(ori_img, a2, 2, (255, 0, 0), 2)
        one = [[pointer_line[0][0], pointer_line[0][1]], [a1[0], a1[1]]]
        two = [[pointer_line[0][0], pointer_line[0][1]], [a2[0], a2[1]]]
        three = [[pointer_line[0][0], pointer_line[0][1]], [pointer_line[1][0], pointer_line[1][1]]]
        logger.debug("angle vectors one=%s two=%s three=%s", one, two, three)

        one = np.array(one)
        two = np.array(two)
        three = np.array(three)

        v1 = one[1] - one[0]
        v2 = two[1] - two[0]
        v3 = three[1] - three[0]

        distance = self.get_distance_point2line(
            [a1[0], a1[1]], [pointer_line[0][0], pointer_line[0][1], pointer_line[1][0], pointer_line[1][1]]
        )
        # print("dis",distance)

        flag = self.judge(pointer_line[0], std_point[0], pointer_line[1])
        # print("flag",flag)

        std_ang = self.angle(v1, v2)
        logger.debug("std angle=%s", std_ang)
        now_ang = self.angle(v1, v3)
        if flag > 0:
            now_ang = 360 - now_ang
        logger.debug("pointer angle=%s", now_ang)

        # calculate value
        ratio = 0.0
        if std_ang != 0:
            ratio = now_ang / std_ang

        logger.debug("angle ratio pointer/full=%.4f", ratio)

        if number is not None and number[0] != "":
            two_value = float(number[0])
        else:
            # Even if number is missing, show ratio
            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.debug:
                ori_img = cv2.putText(ori_img, f"Ratio: {ratio:.2f}", (30, 80), font, 1.0, (0, 255, 255), 2)
                cv2.imshow("result", ori_img)
                cv2.waitKey(0)
            return f"Ratio: {ratio}"

        if std_ang * now_ang != 0:
            value = two_value / std_ang
            value = value * now_ang
        else:
            logger.warning("angle detection failed: std_ang=%s now_ang=%s", std_ang, now_ang)
            return "angle detect error"

        if flag > 0 and distance < 40:
            value = 0.00
            ratio = 0.0  # Correction for zero position
        else:
            value = round(value, 3)

        if self.debug:
            font = cv2.FONT_HERSHEY_SIMPLEX
            ori_img = cv2.putText(ori_img, str(value), (30, 30), font, 1.2, (255, 0, 255), 2)
            ori_img = cv2.putText(ori_img, f"Ratio: {ratio:.2f}", (30, 80), font, 1.0, (0, 255, 255), 2)
            cv2.imshow("result", ori_img)
            cv2.waitKey(0)

        return value

    def calculate_center_from_geometry(self, std_point, pointer_line):
        """
        Calculate center based on the assumption that:
        1. The perpendicular bisector of the two scale points (start and end of dial) passes through the center.
        2. The line extension of the pointer passes through the center.
        intersection of these two lines is the center.
        """
        p1 = np.array(std_point[0])
        p2 = np.array(std_point[1])

        # Midpoint of the two scale points
        mid_scale = (p1 + p2) / 2

        # Vector of the chord connecting the two scale points
        chord_vec = p2 - p1

        # Perpendicular vector to the chord (rotate 90 degrees)
        perp_vec = np.array([-chord_vec[1], chord_vec[0]])

        # Line 1: Perpendicular bisector defined by point 'mid_scale' and direction 'perp_vec'
        # L1(t) = mid_scale + t * perp_vec

        # Line 2: Pointer line defined by point 'pointer_line[0]' and direction 'pointer_vec'
        ptr_p1 = np.array(pointer_line[0])
        ptr_p2 = np.array(pointer_line[1])
        pointer_vec = ptr_p2 - ptr_p1

        # Solve for intersection
        # mid_scale + t * perp_vec = ptr_p1 + u * pointer_vec
        # t * perp_vec - u * pointer_vec = ptr_p1 - mid_scale

        A = np.array([perp_vec, -pointer_vec]).T
        b = ptr_p1 - mid_scale

        try:
            x = np.linalg.solve(A, b)  # x = [t, u]
            center = mid_scale + x[0] * perp_vec
            return (int(center[0]), int(center[1]))
        except np.linalg.LinAlgError:
            return None  # Parallel lines or other singularity

    def get_distance_point2line(self, point, line):
        """
        Args:
            point: [x0, y0]
            line: [x1, y1, x2, y2]
        """
        line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        denom = np.linalg.norm(line_point1 - line_point2)
        if denom < 1e-8:
            return float("inf")
        distance = np.abs(np.cross(vec1, vec2)) / denom
        return distance

    def judge(self, p1, p2, p3):
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p2[0] * p1[1] - p1[0] * p2[1]

        value = A * p3[0] + B * p3[1] + C

        return value

    def angle(self, v1, v2):
        lx = np.sqrt(v1.dot(v1))
        ly = np.sqrt(v2.dot(v2))
        if lx < 1e-8 or ly < 1e-8:
            return 0.0
        cos_angle = v1.dot(v2) / (lx * ly)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        angle = np.arccos(cos_angle)
        angle2 = angle * 360 / 2 / np.pi

        return angle2
