import os
import sys
import uvicorn
import cv2
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(current_dir)
repo_root = os.path.dirname(package_root)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from gauge_read.webui.gauge_logic import GaugeAppModel

# Setup Args
parser = argparse.ArgumentParser(description="Gauge Reader API")
parser.add_argument("--yolo", type=str, default="pretrain/best.pt", help="Path to YOLO weights")
parser.add_argument("--stn", type=str, default="logs/stn/stn_ep50_loss0.0108.pth", help="Path to STN weights")
parser.add_argument("--textnet", type=str, default="pretrain/textgraph_convnext_tiny_100.pth", help="Path to TextNet weights")
parser.add_argument("--host", type=str, default="0.0.0.0", help="API Host")
parser.add_argument("--port", type=int, default=8000, help="API Port")
args = parser.parse_args()

# Initialize App Logic
print("Initializing Gauge Model...")
app_logic = GaugeAppModel()
res = app_logic.load_models(textnet_path=args.textnet, stn_path=args.stn, yolo_path=args.yolo)
print(res)

# API
app = FastAPI(title="Gauge Reader API")


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

    vis_img, val, auto_start, auto_end = app_logic.process_image(image, use_stn=req.use_stn, use_yolo=req.use_yolo)

    if val is None:
        raise HTTPException(status_code=500, detail=f"Inference failed: {vis_img}")  # vis_img contains error msg if val is None

    # Override values if provided
    need_recalc = False
    if req.start_value is not None:
        app_logic.current_start_value = req.start_value
        need_recalc = True

    if req.end_value is not None:
        app_logic.current_end_value = req.end_value
        need_recalc = True

    final_val = val
    if need_recalc:
        final_val = app_logic.recalculate()

    # Get Ratio
    final_ratio = getattr(app_logic, "current_ratio", 0.0)

    return GaugeResponse(
        measure_value=final_val,
        ratio=final_ratio,
        start_value=app_logic.current_start_value,
        end_value=app_logic.current_end_value,
    )


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
