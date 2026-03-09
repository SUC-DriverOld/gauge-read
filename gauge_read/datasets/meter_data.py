import os
import numpy as np
import json
import cv2
import copy
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from gauge_read.utils.config import config as cfg
from gauge_read.utils.stn_transform import STNTransformer
from gauge_read.datasets.synth_gauge import gen_gauge


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None

        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        self.points = np.array(points)

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(Dataset):
    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training

    @staticmethod
    def make_text_region(polygon, mask):
        cv2.fillPoly(mask, [polygon.points.astype(np.int32)], 1)  # make text_region

        return mask

    def get_training_data(self, image, polygons, transcripts, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))

        pointer_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        dail_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        text_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

        if polygons is not None:
            for _, polygon in enumerate(polygons):
                if polygon.text == "1":
                    pointer_mask = self.make_text_region(polygon, pointer_mask)

                if polygon.text == "2":
                    dail_mask = self.make_text_region(polygon, dail_mask)

                if polygon.text == "number":
                    text_mask = self.make_text_region(polygon, text_mask)
                    bboxs = polygon.points.reshape((1, 8))

        train_mask = np.ones(image.shape[:2], np.uint8)
        image = image.transpose(2, 0, 1)

        if not self.is_training:  # test condition
            meta = {"image_id": image_id, "Height": H, "Width": W, "trans": transcripts}

            return image, pointer_mask, dail_mask, text_mask, train_mask, meta

        return image, pointer_mask, dail_mask, text_mask, train_mask, bboxs, transcripts

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()


class MeterDataset(TextDataset):
    def __init__(self, root="./datas", mode="train", mode1="train1", is_training=True, transform=None):
        super().__init__(transform, is_training)
        self.dataset = []
        self.name = []
        image_path = f"{root}/images/"
        mask_path = f"{root}/annotations/{mode}"
        mask_path1 = f"{root}/annotations/{mode1}"

        for image_name in os.listdir(image_path):
            mask_name = image_name.split(".")[0] + ".json"
            self.dataset.append((f"{image_path}/{image_name}", f"{mask_path}/{mask_name}", f"{mask_path1}/{mask_name}"))
            self.name.append(image_name)

        self.annotation_cache = {}

        if cfg.data.get("stn_correction", False):
            # Use CPU for dataset workers to avoid multiprocessing issues with CUDA
            self.stn_transformer = STNTransformer(cfg.data.get("stn_model_path", ""))
        else:
            self.stn_transformer = None

    @staticmethod
    def parse_txt(mask_path, mask_path1):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """

        with open(mask_path, "r") as load_f:
            load_dict = json.load(load_f)

        info = load_dict["shapes"]
        polygons = []
        for i in range(len(info)):
            points = info[i]["points"]
            points = np.array(points).astype(np.int32)
            label = info[i]["label"]
            polygons.append(TextInstance(points, "c", label))

        with open(mask_path1, "r") as load_f:
            load_dict = json.load(load_f)
        info = load_dict["shapes"]

        transcripts = []
        for i in range(len(info)):
            points = info[i]["points"]
            points = np.array(points).astype(np.int32)
            text = info[i]["label"]
            transcripts.append(text)
            label = "number"
            polygons.append(TextInstance(points, "c", label))

        return polygons, transcripts

    def __getitem__(self, item):
        image_path, mask_path, mask_path1 = self.dataset[item]
        idx = self.name[item]

        # Read image data
        image = pil_load_img(image_path)

        cache_key = (mask_path, mask_path1)
        if cache_key in self.annotation_cache:
            polygons, transcripts = copy.deepcopy(self.annotation_cache[cache_key])
        else:
            try:
                polygons, transcripts = self.parse_txt(mask_path, mask_path1)
                self.annotation_cache[cache_key] = (copy.deepcopy(polygons), copy.deepcopy(transcripts))
            except Exception:
                polygons = None
                transcripts = []

        # Apply STN correction
        if self.stn_transformer is not None and polygons is not None:
            image, polygons, _ = self.stn_transformer(image, polygons)

        # if polygons is not None:
        #     print(f"Loaded {len(polygons)} polygons for {idx}")
        # else:
        #     print(f"No polygons for {idx}")

        return self.get_training_data(image, polygons, transcripts, image_id=idx, image_path=image_path)

    def __len__(self):
        return len(self.dataset)


class MeterSyn(Dataset):
    def __init__(self, size=80000, use_homography=True, use_artefacts=True, use_arguments=True):
        self.size = size
        self.use_homography = use_homography
        self.use_artefacts = use_artefacts
        self.use_arguments = use_arguments

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        img, _, _, Minv, center = gen_gauge(self.use_homography, self.use_artefacts, self.use_arguments)

        # 将原图上的圆心坐标归一化到 [0, 1] 区间，原始图像尺寸默认是 512x512
        H, W = img.shape[:2]
        cx, cy = center
        center_norm = np.array([cx / W, cy / H], dtype=np.float32)

        img = np.clip(img, 0, 255)
        img = cv2.resize(img, (224, 224)) / 255.0
        img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img, Minv, center_norm


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
