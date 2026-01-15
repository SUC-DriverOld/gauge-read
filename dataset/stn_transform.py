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
    def __init__(self, model_path, device='cpu'):
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
                if 'model_state_dict' in checkpoint:
                     self.model.load_state_dict(checkpoint['model_state_dict'])
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
            minv = Minv_pred.squeeze(0).cpu().numpy() # This is likely for normalized coords [-1, 1]

        # Convert normalized homography to pixel homography for HxW image
        # Normalization matrix N
        # [ 2/s  0   -1 ]
        # [  0  2/s  -1 ]
        # [  0   0    1 ]
        
        N = np.array([
            [2.0/s, 0, -1],
            [0, 2.0/s, -1],
            [0, 0, 1]
        ])
        
        N_inv = np.linalg.inv(N)
        
        # M_pixel = N_inv @ M_norm @ N
        # Note: If Minv_pred is the Inverse Matrix (Target -> Source), we accept it as is if we use warpPerspective with WARP_INVERSE_MAP
        # Or usually warpPerspective takes H that maps Source -> Target. 
        # The STN usually produces a grid sampler matrix which maps Target(grid) -> Source(sampling points).
        # So Minv_pred is likely T -> S.
        # cv2.warpPerspective(src, M, dsize) uses M to map src(x,y) -> dst(u,v). 
        # So inputs M should be S -> T.
        # If STN outputs T -> S, then we should use flags=cv2.WARP_INVERSE_MAP or invert the matrix.
        
        # Let's verify stn/utils.py logic. 
        # It assumes Minv_pred is used with kornia.warp_perspective.
        # kornia.warp_perspective takes M and warps tensor.
        # "Warps a tensor by the homography matrix."
        # Usually STNs predict theta for affine_grid, which is T -> S.
        # If Minv_pred is T -> S.
        
        # Let's try to assume Minv_pred is T -> S (Inverse Homography).
        # We need S -> T for cv2.warpPerspective.
        # So we might need to invert it.
        
        # However, looking at stn/utils.py again:
        # Minv_pred = Scale_matrix @ Minv_pred @ Scale_matrix_inv.
        # This scales the matrix from normalized to 224x224 pixel space.
        # So Minv_pred was in normalized space.
        
        M_pixel_inv = N_inv @ minv @ N
        
        # We want H for cv2.warpPerspective (Source -> Target).
        # If M_pixel_inv matches T -> S.
        # Then H = inv(M_pixel_inv).
        
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
        
        # Warp canvas instead of original image
        image_warped = cv2.warpPerspective(canvas, H_mat, (m, m))
        
        # Warp polygons
        new_polygons = []
        if polygons is not None:
            for poly in polygons:
                # poly.points is numpy (N, 2)
                # Need to convert to (N, 1, 2) float32 for perspectiveTransform
                pts = poly.points.reshape(-1, 1, 2).astype(np.float32)
                transformed_pts = cv2.perspectiveTransform(pts, H_mat)
                transformed_pts = transformed_pts.reshape(-1, 2)
                
                # Update polygon points
                poly.points = transformed_pts
                new_polygons.append(poly)
                
        return image_warped, new_polygons
