import os
import sys
import argparse
import threading
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(current_dir)
repo_root = os.path.dirname(package_root)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from gauge_read.webui.app_logic import GaugeAppModel
from gauge_read.utils.config import AttrDict
from gauge_read.utils.logger import logger

app = FastAPI(title="Gauge Reader API")
_infer_lock = threading.Lock()
_app_logic: Optional[GaugeAppModel] = None
_cfg: Optional[AttrDict] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Gauge Reader API")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--yolo", type=str, default=None, help="Path to YOLO weights")
    parser.add_argument("--stn", type=str, default=None, help="Path to STN weights")
    parser.add_argument("--textnet", type=str, default=None, help="Path to TextNet weights")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API Host")
    parser.add_argument("--port", type=int, default=8000, help="API Port")
    return parser.parse_args()


def _resolve_model_paths(cfg, yolo_path=None, stn_path=None, textnet_path=None):
    yolo = yolo_path or cfg.predict.get("yolo_model_path", "pretrain/best.pt")
    stn = stn_path if stn_path is not None else cfg.data.get("stn_model_path", "")
    textnet = textnet_path or cfg.predict.get("model_path", "")
    return yolo, stn, textnet


def _get_cfg():
    global _cfg
    if _cfg is None:
        _cfg = AttrDict(AttrDict.DEFAULT_CONFIG_PATH)
        logger.info("API default configuration loaded: %s", AttrDict.DEFAULT_CONFIG_PATH)
    return _cfg


def init_app_logic(yolo_path=None, stn_path=None, textnet_path=None):
    global _app_logic
    cfg = _get_cfg()
    yolo, stn, textnet = _resolve_model_paths(cfg, yolo_path, stn_path, textnet_path)

    logger.info("Initializing Gauge Model for API: textnet=%s, stn=%s, yolo=%s", textnet, stn or "disabled", yolo)
    _app_logic = GaugeAppModel(cfg)
    _app_logic.load_models(textnet_path=textnet, stn_path=stn, yolo_path=yolo)
    logger.info("Gauge API model initialization completed")


@app.on_event("startup")
def startup_event():
    # Keep module-mode startup working: uvicorn gauge_read.scripts.api:app
    if _app_logic is None:
        logger.info("FastAPI startup event triggered; initializing model state")
        init_app_logic()


class GaugeRequest(BaseModel):
    image_path: str
    use_yolo: bool = False
    use_stn: bool = False
    start_value: Optional[float] = None
    end_value: Optional[float] = None


class GaugeResponse(BaseModel):
    measure_value: float
    ratio: float
    start_value: float
    end_value: float
    status: str = "success"


@app.post("/predict", response_model=GaugeResponse)
def predict(req: GaugeRequest):
    if _app_logic is None:
        logger.error("API predict called before model initialization")
        raise HTTPException(status_code=500, detail="Model is not initialized")

    logger.info(
        "API predict request received: image_path=%s, use_yolo=%s, use_stn=%s, start_override=%s, end_override=%s",
        req.image_path,
        req.use_yolo,
        req.use_stn,
        req.start_value,
        req.end_value,
    )

    if not os.path.exists(req.image_path):
        logger.warning("API request image path not found: %s", req.image_path)
        raise HTTPException(status_code=400, detail=f"Image path not found: {req.image_path}")

    # Read Image
    image = cv2.imread(req.image_path)
    if image is None:
        logger.warning("API failed to decode image: %s", req.image_path)
        raise HTTPException(status_code=400, detail="Failed to read image")

    logger.debug("API input image loaded with shape=%s", image.shape)

    # Process
    # process_image returns: display_img, val, start_val, end_val
    # Note: process_image sets current_end_value from OCR if not overridden
    # But currently process_image doesn't accept start/end overrides arguments.
    # It resets start=0 and end=OCR.
    # We run inference first to get OCR result (if needed) and std points.

    # GaugeAppModel contains mutable current_* state, so use a lock to avoid cross-request contamination.
    with _infer_lock:
        logger.debug("API inference lock acquired")
        vis_img, val, auto_start, auto_end = _app_logic.process_image(image, use_stn=req.use_stn, use_yolo=req.use_yolo)

        if val is None:
            logger.error("API inference returned no value: %s", vis_img)
            raise HTTPException(status_code=500, detail=f"Inference failed: {vis_img}")

        # Override values if provided.
        need_recalc = False
        if req.start_value is not None:
            _app_logic.current_start_value = req.start_value
            need_recalc = True
            logger.info("API start value override applied: %s", req.start_value)

        if req.end_value is not None:
            _app_logic.current_end_value = req.end_value
            need_recalc = True
            logger.info("API end value override applied: %s", req.end_value)

        final_val = val
        if need_recalc:
            logger.info("API recalculating result after manual overrides")
            final_val = _app_logic.recalculate()

        final_ratio = getattr(_app_logic, "current_ratio", 0.0)

        logger.info(
            "API inference completed: measure_value=%s, ratio=%s, start_value=%s, end_value=%s",
            final_val,
            final_ratio,
            _app_logic.current_start_value,
            _app_logic.current_end_value,
        )

        return GaugeResponse(
            measure_value=final_val,
            ratio=final_ratio,
            start_value=_app_logic.current_start_value,
            end_value=_app_logic.current_end_value,
        )


if __name__ == "__main__":
    args = parse_args()
    _cfg = AttrDict(args.config or AttrDict.DEFAULT_CONFIG_PATH)
    logger.info(
        "Starting Gauge Reader API: host=%s, port=%s, config=%s",
        args.host,
        args.port,
        args.config or AttrDict.DEFAULT_CONFIG_PATH,
    )
    init_app_logic(yolo_path=args.yolo, stn_path=args.stn, textnet_path=args.textnet)
    uvicorn.run(app, host=args.host, port=args.port)
