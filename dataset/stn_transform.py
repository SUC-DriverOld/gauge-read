import torch
import cv2
import numpy as np
import sys
import os

# Ensure we can import stn.stn_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from stn.stn_model import STNModel
except ImportError:
    # Fallback or error handling if stn folder is not found or not in path
    print("Warning: Could not import STNModel. STN correction will be disabled.")
    STNModel = None


class STNTransformer:
    def __init__(self, model_path, device="cpu"):
        if STNModel is None:
            self.model = None
            return

        self.device = device
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

    def process_image(self, image):
        """
        Pad to square (black filled), resize to 224x224, normalize, convert to tensor
        """
        h, w = image.shape[:2]
        m = max(h, w)
        if len(image.shape) == 3:
            canvas = np.zeros((m, m, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((m, m), dtype=image.dtype)
        canvas[:h, :w] = image

        img_resized = cv2.resize(canvas, (224, 224))
        # Convert to tensor: (H, W, C) -> (C, H, W) -> Batch
        img_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor

    def get_homography_matrix(self, image):
        if self.model is None:
            return None

        h, w = image.shape[:2]
        s = max(h, w)
        img_tensor = self.process_image(image)

        with torch.no_grad():
            Minv_pred, _ = self.model(img_tensor)
            # Minv_pred is [1, 3, 3]
            minv = Minv_pred.squeeze(0).cpu().numpy()  # This is likely for normalized coords [-1, 1]

        N = np.array([[2.0 / s, 0, -1], [0, 2.0 / s, -1], [0, 0, 1]])

        N_inv = np.linalg.inv(N)

        M_pixel_inv = N_inv @ minv @ N
        try:
            H = np.linalg.inv(M_pixel_inv)
        except np.linalg.LinAlgError:
            return None

        return H

    def __call__(self, image, polygons):
        if self.model is None:
            return image, polygons

        H_mat = self.get_homography_matrix(image)
        if H_mat is None:
            return image, polygons

        h, w = image.shape[:2]
        m = max(h, w)

        # Create padded canvas
        if len(image.shape) == 3:
            canvas = np.zeros((m, m, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((m, m), dtype=image.dtype)
        canvas[:h, :w] = image

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

        return image_warped, new_polygons
