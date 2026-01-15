import os
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from SynGagues import gen_gauge


class ClockSyn(Dataset):
    def __init__(self, size=80000, use_homography=True, use_artefacts=True, use_arguments=True):
        self.size = size
        self.use_homography = use_homography
        self.use_artefacts = use_artefacts
        self.use_arguments = use_arguments

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        img, _, _, Minv = gen_gauge(self.use_homography, self.use_artefacts, self.use_arguments)
        img = np.clip(img, 0, 255)
        img = cv2.resize(img, (224, 224)) / 255.0
        img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img, Minv


class STNTest(Dataset):
    def __init__(self, root_dir, size=224):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        self.transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img
