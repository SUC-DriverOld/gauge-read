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
from gauge_read.utils.config import config as cfg, load_config

app = FastAPI(title="Gauge Reader API")
_infer_lock = threading.Lock()
_app_logic: Optional[GaugeAppModel] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Gauge Reader API")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--yolo", type=str, default=None, help="Path to YOLO weights")
    parser.add_argument("--stn", type=str, default=None, help="Path to STN weights")
    parser.add_argument("--textnet", type=str, default=None, help="Path to TextNet weights")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API Host")
    parser.add_argument("--port", type=int, default=8000, help="API Port")
    return parser.parse_args()


def _resolve_model_paths(yolo_path=None, stn_path=None, textnet_path=None):
    yolo = yolo_path or cfg.predict.get("yolo_model_path", "pretrain/best.pt")
    stn = stn_path if stn_path is not None else cfg.data.get("stn_model_path", "")
    textnet = textnet_path or cfg.predict.get("model_path", "")
    return yolo, stn, textnet


def init_app_logic(yolo_path=None, stn_path=None, textnet_path=None):
    global _app_logic
    yolo, stn, textnet = _resolve_model_paths(yolo_path, stn_path, textnet_path)

    print("Initializing Gauge Model...")
    _app_logic = GaugeAppModel()
    _app_logic.load_models(textnet_path=textnet, stn_path=stn, yolo_path=yolo)


@app.on_event("startup")
def startup_event():
    # Keep module-mode startup working: uvicorn gauge_read.scripts.api:app
    if _app_logic is None:
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
        raise HTTPException(status_code=500, detail="Model is not initialized")

    if not os.path.exists(req.image_path):
        raise HTTPException(status_code=400, detail=f"Image path not found: {req.image_path}")

    # Read Image
    image = cv2.imread(req.image_path)
    if image is None:
        raise HTTPException(status_code=400, detail="Failed to read image")

    # Process
    # process_image returns: display_img, val, start_val, end_val
    # Note: process_image sets current_end_value from OCR if not overridden
    # But currently process_image doesn't accept start/end overrides arguments.
    # It resets start=0 and end=OCR.
    # We run inference first to get OCR result (if needed) and std points.

    # GaugeAppModel contains mutable current_* state, so use a lock to avoid cross-request contamination.
    with _infer_lock:
        vis_img, val, auto_start, auto_end = _app_logic.process_image(image, use_stn=req.use_stn, use_yolo=req.use_yolo)

        if val is None:
            raise HTTPException(status_code=500, detail=f"Inference failed: {vis_img}")

        # Override values if provided.
        need_recalc = False
        if req.start_value is not None:
            _app_logic.current_start_value = req.start_value
            need_recalc = True

        if req.end_value is not None:
            _app_logic.current_end_value = req.end_value
            need_recalc = True

        final_val = val
        if need_recalc:
            final_val = _app_logic.recalculate()

        final_ratio = getattr(_app_logic, "current_ratio", 0.0)

        return GaugeResponse(
            measure_value=final_val,
            ratio=final_ratio,
            start_value=_app_logic.current_start_value,
            end_value=_app_logic.current_end_value,
        )


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        load_config(args.config)
    init_app_logic(yolo_path=args.yolo, stn_path=args.stn, textnet_path=args.textnet)
    uvicorn.run(app, host=args.host, port=args.port)
