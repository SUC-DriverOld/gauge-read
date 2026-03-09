import torch
import cv2
import numpy as np
import os
from gauge_read.models.stn import STNModel


class STNTransformer:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self._norm_cache = {}
        self.model = STNModel(pretrained=False)

        if os.path.exists(model_path):
            try:
                # Load state dict
                checkpoint = torch.load(model_path, map_location=device)
                # Handle case where checkpoint might be model state or dict
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Error loading STN model: {e}")
                self.model = None
                return
        else:
            print(f"STN model path not found: {model_path}")
            self.model = None
            return

        self.model.to(device)
        self.model.eval()

    def _get_norm_mats(self, s):
        if s not in self._norm_cache:
            N = np.array([[2.0 / s, 0, -1], [0, 2.0 / s, -1], [0, 0, 1]], dtype=np.float32)
            self._norm_cache[s] = (N, np.linalg.inv(N))
        return self._norm_cache[s]

    @staticmethod
    def _pad_to_square(image):
        h, w = image.shape[:2]
        m = max(h, w)
        if len(image.shape) == 3:
            canvas = np.zeros((m, m, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((m, m), dtype=image.dtype)
        canvas[:h, :w] = image
        return canvas, h, w, m

    def process_image(self, image):
        """
        Pad to square (black filled), resize to 224x224, normalize, convert to tensor
        """
        canvas, _, _, _ = self._pad_to_square(image)

        img_resized = cv2.resize(canvas, (224, 224))
        # Convert to tensor: (H, W, C) -> (C, H, W) -> Batch
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor

    def get_homography_matrix(self, image):
        if self.model is None:
            return None, None

        s = max(image.shape[:2])
        img_tensor = self.process_image(image)

        with torch.no_grad():
            Minv_pred, _, pred_center = self.model(img_tensor)
            # Minv_pred is [1, 3, 3]
            minv = Minv_pred.squeeze(0).cpu().numpy()  # This is likely for normalized coords [-1, 1]
            center_norm = pred_center.squeeze(0).cpu().numpy()

        N, N_inv = self._get_norm_mats(s)

        M_pixel_inv = N_inv @ minv @ N
        try:
            H = np.linalg.inv(M_pixel_inv)
        except np.linalg.LinAlgError:
            return None, None

        # Calculate center in padded pixel coordinates
        center_pixel = np.array([center_norm[0] * s, center_norm[1] * s], dtype=np.float32)

        return H, center_pixel

    def __call__(self, image, polygons):
        if self.model is None:
            return image, polygons, None

        H_mat, center_pixel = self.get_homography_matrix(image)
        if H_mat is None:
            return image, polygons, None

        canvas, _, _, m = self._pad_to_square(image)

        # Calculate new bounds to encompass the warped image
        corners = np.array([[0, 0], [m, 0], [m, m], [0, m]], dtype=np.float32).reshape(-1, 1, 2)
        corners_transformed = cv2.perspectiveTransform(corners, H_mat)
        x_min = corners_transformed[:, 0, 0].min()
        x_max = corners_transformed[:, 0, 0].max()
        y_min = corners_transformed[:, 0, 1].min()
        y_max = corners_transformed[:, 0, 1].max()

        # Dimensions of the bounding box
        width = x_max - x_min
        height = y_max - y_min

        # Determine the side length of the square canvas just enough to fit
        side = int(np.ceil(max(width, height)))

        # Compute translation to center the warped image content in the square canvas
        tx = (side - width) / 2.0 - x_min
        ty = (side - height) / 2.0 - y_min

        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        H_new = T @ H_mat

        # Warp canvas with new square size and centered transformation
        image_warped = cv2.warpPerspective(canvas, H_new, (side, side))

        # Warp center
        warped_center = None
        if center_pixel is not None:
            c_pts = np.array([[[center_pixel[0], center_pixel[1]]]], dtype=np.float32)
            c_pts_warped = cv2.perspectiveTransform(c_pts, H_new)
            warped_center = c_pts_warped[0][0]

        # Warp polygons
        new_polygons = []
        if polygons is not None:
            for poly in polygons:
                # poly.points is numpy (N, 2)
                # Need to convert to (N, 1, 2) float32 for perspectiveTransform
                pts = poly.points.reshape(-1, 1, 2).astype(np.float32)
                transformed_pts = cv2.perspectiveTransform(pts, H_new)
                transformed_pts = transformed_pts.reshape(-1, 2)

                # Update polygon points
                poly.points = transformed_pts
                new_polygons.append(poly)

        return image_warped, new_polygons, warped_center
