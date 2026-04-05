import argparse
import os

import cv2
import numpy as np
import torch

from gauge_read.models.textnet import TextNet
from gauge_read.utils.stn_transform import STNTransformer
from gauge_read.utils.augmentation import BaseTransform
from gauge_read.utils.config import AttrDict
from gauge_read.utils.converter import StringLabelConverter
from gauge_read.utils.logger import logger
from gauge_read.utils.tools import to_device
from gauge_read.utils.reader import MeterReader, TextDetector, YOLODetector


def main(args, cfg):
    stn_transformer = None
    if cfg.predict.get("use_stn", False):
        stn_model_path = cfg.data.get("stn_model_path", "")
        logger.info("Initializing STN for inference from %s", stn_model_path)
        stn_transformer = STNTransformer(stn_model_path, device=cfg.system.device)

    predict_dir = args.data_dir or cfg.predict.get("data_dir", "datas/demo")
    logger.info("Starting offline inference: predict_dir=%s", predict_dir)

    if not os.path.isdir(predict_dir):
        logger.error("Predict directory not found: %s", predict_dir)
        raise NotADirectoryError(f"Predict directory not found: {predict_dir}")

    model = TextNet(is_training=False, backbone=cfg.model.net, cfg=cfg)
    model_path = cfg.predict.model_path
    if not os.path.exists(model_path):
        logger.error("Inference model checkpoint not found: %s", model_path)
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Please update `predict.model_path` in gauge_read/configs/config.yaml or provide a valid config file."
        )
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    model = model.to(cfg.system.device)
    logger.info("Inference TextNet loaded and moved to device=%s", cfg.system.device)
    converter = StringLabelConverter()

    det = YOLODetector(cfg=cfg)
    detector = TextDetector(model)
    meter = MeterReader(debug=True)
    transform = BaseTransform(size=cfg.data.test_size, mean=cfg.model.means, std=cfg.model.stds)

    image_list = os.listdir(predict_dir)
    logger.info("Found %s files in predict directory", len(image_list))
    for index in image_list:
        logger.info("Processing inference image: %s", index)
        image_path = os.path.join(predict_dir, index)
        image = cv2.imread(image_path)

        if image is None:
            logger.warning("Failed to load image from %s; skipping", image_path)
            continue

        cv2.imshow("det1", image)
        cv2.waitKey(0)

        if cfg.predict.get("use_yolo", False):
            _, _, _, meter_list = det.detect(image, index)
        else:
            meter_list = [image]

        if len(meter_list) == 0:
            logger.warning("No meter detected for image %s", index)
            continue

        logger.info("Inference image %s yielded %s meter candidates", index, len(meter_list))

        for meter_img in meter_list:
            predicted_center = None
            if stn_transformer is not None:
                meter_img, _, predicted_center = stn_transformer(meter_img, None)
                logger.debug("STN processed meter candidate, predicted_center=%s", predicted_center)

            cv2.imshow("det", meter_img)
            cv2.waitKey(0)

            norm_img, _ = transform(meter_img)
            norm_img = norm_img.transpose(2, 0, 1)
            norm_img = torch.from_numpy(norm_img).unsqueeze(0)
            norm_img = to_device(norm_img, device=cfg.system.device)
            try:
                output = detector.detect1(norm_img)
            except Exception as _:
                logger.exception("Detection error during offline inference for image %s", index)
                continue

            pointer_pred, dail_pred, text_pred, preds, std_points, aux_map = (
                output["pointer"],
                output["dail"],
                output["text"],
                output["reco"],
                output["std"],
                output["aux"],
            )

            if aux_map is not None:
                cv2.imshow("aux_blackhat", aux_map)

            pred, preds_size = preds
            if pred is not None:
                _, pred = pred.max(2)
                pred = pred.transpose(1, 0).contiguous().view(-1)
                pred_transcripts = converter.decode(pred.data, preds_size.data, raw=False)
                pred_transcripts = [pred_transcripts] if isinstance(pred_transcripts, str) else pred_transcripts
            else:
                pred_transcripts = None

            logger.info(
                "Offline inference result for %s: has_pointer=%s, has_text=%s, has_std_points=%s, transcript=%s",
                index,
                pointer_pred is not None,
                text_pred is not None,
                std_points is not None,
                pred_transcripts,
            )

            img_show = norm_img[0].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * np.array(cfg.model.stds) + np.array(cfg.model.means)) * 255).astype(np.uint8)

            meter(
                img_show,
                pointer_pred,
                dail_pred,
                text_pred,
                pred_transcripts,
                std_points,
                getattr(stn_transformer, "last_center", predicted_center),
            )

    logger.info("Offline inference finished for directory %s", predict_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauge inference")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="Path to YAML config file. If omitted, default config is used."
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing images to predict. Overrides config/default path.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("WebUI console log level set to DEBUG")

    cfg = AttrDict(args.config or AttrDict.DEFAULT_CONFIG_PATH)
    logger.info("Launching offline inference with config=%s", args.config or AttrDict.DEFAULT_CONFIG_PATH)
    cfg.print_config()

    main(args, cfg)
