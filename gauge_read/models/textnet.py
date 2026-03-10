import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from gauge_read.models.convnext import ConvNeXtTiny
from gauge_read.models.crnn import CRNN
from gauge_read.utils.roi import batch_roi_transform
from gauge_read.utils.converter import keys
from gauge_read.utils.tools import order_points


class TorchBlackHatModule(nn.Module):
    def __init__(self, kernel_size=15):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def get_gray(self, x):
        # x is (B, 3, H, W). Assumed to be somewhat normalized.
        # Simple average to get intensity if color weights aren't critical
        return torch.mean(x, dim=1, keepdim=True)

    def forward(self, x):
        # Helper for dilation with replicate pad to avoid border artifacts
        def morph_dilation(tensor):
            padded = F.pad(tensor, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
            return F.max_pool2d(padded, self.kernel_size, stride=1, padding=0)

        def morph_erosion(tensor):
            return -morph_dilation(-tensor)

        # 1. Convert to Gray
        gray = self.get_gray(x)

        # 2. Black Hat: Closed - Original
        # Closing = Erosion(Dilation(img))
        dilated = morph_dilation(gray)
        closed = morph_erosion(dilated)

        black_hat = closed - gray

        # 3. Expand to 3 channels for backbone compatibility
        return black_hat  # Returns 1 channel (B, 1, H, W)


class UpBlok(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class FPN(nn.Module):
    def __init__(self, backbone="convnext_tiny", is_training=True, use_multimodal=False):
        super().__init__()

        self.is_training = is_training
        self.backbone_name = backbone
        self.class_channel = 6
        self.reg_channel = 2

        input_channels = 4 if use_multimodal else 3

        if backbone != "convnext_tiny":
            raise ValueError(f"Unsupported TextNet backbone: {backbone}. Only 'convnext_tiny' is supported.")

        self.backbone = ConvNeXtTiny(pretrain=True, input_channels=input_channels)

        # ConvNeXt-tiny channels for (C1, C2, C3, C4, C5) are (96, 96, 192, 384, 768).
        self.deconv5 = nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1)
        self.merge4 = UpBlok(384 + 256, 128)
        self.merge3 = UpBlok(192 + 128, 64)
        self.merge2 = UpBlok(96 + 64, 32)
        self.merge1 = UpBlok(96 + 32, 32)

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.backbone(x)

        # 4. FPN Upsampling
        up5 = self.deconv5(C5)
        up5 = F.relu(up5)
        up4 = self.merge4(C4, up5)
        up4 = F.relu(up4)
        up3 = self.merge3(C3, up4)
        up3 = F.relu(up3)
        up2 = self.merge2(C2, up3)
        up2 = F.relu(up2)
        up1 = self.merge1(C1, up2)

        return up1, up2, up3, up4, up5


class TextNet(nn.Module):
    def __init__(self, backbone="convnext_tiny", is_training=True, cfg=None):
        super().__init__()

        model_cfg = cfg.model if cfg is not None and "model" in cfg else {}
        self.use_multimodal = model_cfg.get("use_multimodal", False)
        print(f"TextNet Multimodal Status: {self.use_multimodal}")

        self.is_training = is_training
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, self.is_training, use_multimodal=self.use_multimodal)

        self.out_channel = 3
        self.predict = nn.Sequential(nn.Conv2d(32, self.out_channel, kernel_size=1, stride=1, padding=0))

        num_class = len(keys) + 1
        self.recognizer = Recognizer(num_class, nc=4 if self.use_multimodal else 3)

        if self.use_multimodal:
            self.blackhat_gen = TorchBlackHatModule()
        else:
            self.blackhat_gen = None

    def load_model(self, model_path):
        print("Loading from {}".format(model_path))
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict["model"])

    def _prepare_backbone_input(self, x_input):
        if self.use_multimodal:
            with torch.no_grad():
                aux = self.blackhat_gen(x_input)
            return torch.cat([x_input, aux], dim=1), aux
        return x_input, None

    @staticmethod
    def _mask_to_contours_and_centers(mask):
        edges = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            centers.append((int(rect[0][0]), int(rect[0][1])))
        return contours, centers

    def _run_recognizer_from_box(self, feat_map, contour):
        rect_points = cv2.boxPoints(cv2.minAreaRect(contour))
        bboxes = rect_points.astype(np.int32)
        bboxes = order_points(bboxes)
        boxes = bboxes.reshape(1, 8)
        mapping = np.array([0])
        rois = batch_roi_transform(feat_map, boxes[:, :8], mapping)
        preds = self.recognizer(rois)
        preds_size = torch.LongTensor([preds.size(0)] * int(preds.size(1)))
        return preds, preds_size

    def forward(self, x_input, boxes=None, mapping=None):
        x, _ = self._prepare_backbone_input(x_input)

        up1, up2, up3, up4, up5 = self.fpn(x)
        predict_out = self.predict(up1)

        rois = batch_roi_transform(x, boxes[:, :8], mapping)

        preds = self.recognizer(rois)
        preds_size = torch.LongTensor([preds.size(0)] * int(preds.size(1)))
        preds_size = preds_size.to(x.device)

        return predict_out, (preds, preds_size)

    def forward_test(self, x_input):
        x, aux = self._prepare_backbone_input(x_input)

        up1, up2, up3, up4, up5 = self.fpn(x)
        output = self.predict(up1)

        # Vectorized thresholding for three segmentation maps.
        probs = torch.sigmoid(output[0])
        th = torch.tensor([0.5, 0.5, 0.7], device=probs.device).view(3, 1, 1)
        binary = (probs > th).to(torch.uint8).cpu().numpy()
        pointer_pred, dail_pred, text_pred = binary[0], binary[1], binary[2]

        dail_label = self.filter(dail_pred, n=30)
        text_label = self.filter(text_pred)

        dail_contours, std_point = self._mask_to_contours_and_centers(dail_label)
        text_contours, ref_point = self._mask_to_contours_and_centers(text_label)

        if len(std_point) == 0:
            return pointer_pred, dail_label, text_label, (None, None), None, None

        if len(std_point) < 2:
            if len(ref_point) == 0:
                return pointer_pred, dail_label, text_label, (None, None), None, None
            std_point.append(ref_point[0])
        else:
            if std_point[0][1] >= std_point[1][1]:
                pass
            else:
                std_point[0], std_point[1] = std_point[1], std_point[0]

        if len(text_contours) != 0:
            # Select text contour nearest to std end point.
            ref_arr = np.array(ref_point, dtype=np.float32)
            target = np.array(std_point[1], dtype=np.float32)
            dists = np.sum((ref_arr - target) ** 2, axis=1)
            index = int(np.argmin(dists))
            preds, preds_size = self._run_recognizer_from_box(x, text_contours[index])
        else:
            preds = None
            preds_size = None

        # Prepare auxiliary map for visualization if it exists
        aux_map = None
        if self.use_multimodal and aux is not None:
            # aux is (B, 1, H, W). Take 0th in batch, 0th channel
            aux_map = aux[0, 0].cpu().numpy()
            aux_map = (aux_map * 255).astype(np.uint8)

        return pointer_pred, dail_label, text_label, (preds, preds_size), std_point, aux_map

    def filter(self, image, n=30):
        image = image.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
        if num_labels <= 1:
            return image
        keep = stats[:, cv2.CC_STAT_AREA] >= n
        keep[0] = False  # ignore background
        return keep[labels].astype(np.uint8)


class Recognizer(nn.Module):
    def __init__(self, nclass, nc=3):
        super().__init__()
        self.crnn = CRNN(32, nc, nclass, 256)

    def forward(self, rois):
        return self.crnn(rois)
