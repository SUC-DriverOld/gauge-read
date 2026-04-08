import argparse
import base64
import csv
import io
import os
import shutil
import sys
import tempfile
import threading
import uuid
import webbrowser
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image
from pydantic import BaseModel

current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(current_dir)
repo_root = os.path.dirname(package_root)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from gauge_read.utils.app_logic import GaugeApp
from gauge_read.utils.config import AttrDict
from gauge_read.utils.logger import logger


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INSTRUCTIONS = [
    "先选择仪表读数模型、STN 矫正模型和 YOLO 检测模型，再点击加载模型。",
    "单图推理支持上传图片、启用或关闭 STN 与 YOLO，并展示识别结果与可编辑点位。",
    "如果结果不准，可以选择修正模式后点击图片，也可以直接修改起始值与结束值。",
    "批量推理支持输入图片目录，自动输出结果预览、CSV 与结果图片 ZIP 下载。",
]


class NativeGaugeApp(GaugeApp):
    def update_point(self, point_type, x, y):
        if self.textnet is None or self.detector is None:
            return None, "模型未加载"
        if self.current_image is None:
            return None, "请先运行推理"

        x, y = int(x), int(y)
        if point_type == "start":
            if not self.current_std_points:
                self.current_std_points = [(0, 0), (0, 0)]
            self.current_std_points[0] = (x, y)
        elif point_type == "end":
            if len(self.current_std_points) < 2:
                self.current_std_points.append((0, 0))
            self.current_std_points[1] = (x, y)
        elif point_type == "pointer_tip":
            if not self.current_pointer_line:
                self.current_pointer_line = [(0, 0), (0, 0)]
            self.current_pointer_line[1] = (x, y)
        elif point_type == "pointer_root":
            if not self.current_pointer_line:
                self.current_pointer_line = [(0, 0), (0, 0)]
            self.current_pointer_line[0] = (x, y)
        elif point_type == "center":
            self.current_center = (x, y)

        return self.draw_visualization(), self.recalculate()

    def update_start_val(self, text):
        if self.textnet is None or self.detector is None:
            return None, "模型未加载"
        if self.current_image is None:
            return None, "请先运行推理"

        try:
            self.current_start_value = float(text)
        except ValueError:
            return None, "起始值输入无效"
        self.current_ocr_error = False
        return self.draw_visualization(), self.recalculate()

    def update_end_val(self, text):
        if self.textnet is None or self.detector is None:
            return None, "模型未加载"
        if self.current_image is None:
            return None, "请先运行推理"

        try:
            self.current_end_value = float(text)
        except ValueError:
            return None, "结束值输入无效"
        self.current_ocr_error = False
        return self.draw_visualization(), self.recalculate()


class BatchInferenceJSON(NativeGaugeApp):
    def process_directory(self, input_dir, use_stn=True, use_yolo=False):
        if self.textnet is None:
            raise RuntimeError("模型未加载")

        input_dir = (input_dir or "").strip()
        if not input_dir:
            raise FileNotFoundError("请输入图片文件夹路径")
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"目录不存在: {input_dir}")

        image_paths = sorted(
            [
                os.path.join(input_dir, name)
                for name in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, name)) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            ]
        )
        if not image_paths:
            raise FileNotFoundError(f"目录中未找到图片: {input_dir}")

        rows = []
        for image_path in image_paths:
            logger.info("web batch inference processing file: %s", image_path)
            try:
                with Image.open(image_path) as pil_image:
                    rgb_image = pil_image.convert("RGB")
                    _, reading, start_val, end_val = self.process_image(rgb_image, use_stn, use_yolo)
                result_image = self.draw_visualization_with_value()
                rows.append(
                    {
                        "filename": os.path.basename(image_path),
                        "thumbnail": image_to_data_url(result_image, max_width=220),
                        "full_image": image_to_data_url(result_image),
                        "start": format_metric(start_val),
                        "end": format_metric(end_val),
                        "ratio": format_metric(self.current_ratio),
                        "reading": format_metric(reading),
                        "download_image": result_image,
                    }
                )
            except Exception as exc:
                fallback = np.full((120, 180, 3), 245, dtype=np.uint8)
                cv2.putText(fallback, "ERROR", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 50, 50), 2)
                rows.append(
                    {
                        "filename": os.path.basename(image_path),
                        "thumbnail": image_to_data_url(fallback, max_width=220),
                        "full_image": image_to_data_url(fallback),
                        "start": "-",
                        "end": "-",
                        "ratio": "-",
                        "reading": f"推理失败: {exc}",
                        "download_image": fallback,
                    }
                )

        return rows


class LoadModelsPayload(BaseModel):
    model_path: Optional[str] = None
    stn_path: Optional[str] = None
    yolo_path: Optional[str] = None


class UpdatePointPayload(BaseModel):
    mode: str
    x: float
    y: float


class UpdateValuePayload(BaseModel):
    field: str
    value: str


class BatchJobPayload(BaseModel):
    input_dir: str
    use_stn: bool = True
    use_yolo: bool = True


app = FastAPI(title="Gauge Read Web")
_infer_lock = threading.Lock()
_cfg: Optional[AttrDict] = None
_app_logic: Optional[NativeGaugeApp] = None
_download_cache: dict[str, dict[str, str]] = {}
_batch_jobs: dict[str, dict] = {}


def cleanup_runtime_cache():
    cache_root = Path(repo_root) / ".cache" / "web_runtime"
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)


cleanup_runtime_cache()


def get_cfg():
    global _cfg
    if _cfg is None:
        resolved = AttrDict.DEFAULT_CONFIG_PATH
        _cfg = AttrDict(resolved)
        logger.info("Web default configuration loaded: %s", resolved)
    return _cfg


def get_app_logic():
    global _app_logic
    if _app_logic is None:
        _app_logic = NativeGaugeApp(get_cfg())
    return _app_logic


def get_model_files(directory):
    if not os.path.exists(directory):
        return []
    valid_ext = {".pt", ".pth"}
    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in valid_ext:
            files.append(path)
    return sorted(files)


def default_model_options():
    meter_dir = os.path.join(repo_root, "pretrain", "meter")
    stn_dir = os.path.join(repo_root, "pretrain", "stn")
    yolo_dir = os.path.join(repo_root, "pretrain", "yolo")
    os.makedirs(meter_dir, exist_ok=True)
    os.makedirs(stn_dir, exist_ok=True)
    os.makedirs(yolo_dir, exist_ok=True)
    return {
        "model_options": get_model_files(meter_dir),
        "stn_options": get_model_files(stn_dir),
        "yolo_options": get_model_files(yolo_dir),
    }


def resolve_reader_config_path(model_path):
    if not model_path:
        return None

    model_file = Path(model_path)
    candidate = model_file.with_suffix(".yaml")
    if candidate.is_file():
        return str(candidate)
    return None


def build_cfg_for_reader_model(model_path=None, stn_path=None, yolo_path=None):
    config_path = resolve_reader_config_path(model_path)
    if config_path:
        cfg = AttrDict(config_path)
        logger.info("Web config matched reader model: %s -> %s", model_path, config_path)
    else:
        config_path = AttrDict.DEFAULT_CONFIG_PATH
        cfg = AttrDict(config_path)
        logger.info("Web config fallback to default for reader model %s: %s", model_path, config_path)

    if model_path:
        cfg.predict.model_path = model_path
    if stn_path is not None:
        cfg.predict.stn_model_path = stn_path
    if yolo_path is not None:
        cfg.predict.yolo_model_path = yolo_path
    return cfg, config_path


def reset_app_logic(cfg):
    global _cfg, _app_logic
    _cfg = cfg
    _app_logic = NativeGaugeApp(cfg)
    return _app_logic


def image_to_png_bytes(image):
    image_np = np.asarray(image)
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(image_np.astype(np.uint8))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_to_data_url(image, max_width=None):
    image_np = np.asarray(image)
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(image_np.astype(np.uint8))
    if max_width and pil_image.width > max_width:
        scale = max_width / float(pil_image.width)
        target_size = (max_width, max(1, int(pil_image.height * scale)))
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def format_metric(value):
    if isinstance(value, (int, float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def build_timestamped_filename(prefix, suffix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{suffix}"


def state_payload(result_image=None, reading=None):
    logic = get_app_logic()
    image = result_image if result_image is not None else logic.draw_visualization()
    reading_value = logic.current_value if reading is None else reading
    debug_image = getattr(logic, "current_debug_image", None)
    return {
        "result_image": image_to_data_url(image),
        "debug_image": image_to_data_url(debug_image) if debug_image is not None else None,
        "reading": format_metric(reading_value),
        "start_value": format_metric(logic.current_start_value),
        "end_value": format_metric(logic.current_end_value),
        "ratio": format_metric(logic.current_ratio),
        "ocr_error": bool(getattr(logic, "current_ocr_error", False)),
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
    }


def register_download(file_path, filename, media_type):
    file_id = uuid.uuid4().hex
    _download_cache[file_id] = {"path": file_path, "filename": filename, "media_type": media_type}
    return f"/api/download/{file_id}"


def get_runtime_temp_root():
    runtime_root = Path(repo_root) / ".cache" / "web_runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root


def create_csv_file(rows):
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=get_runtime_temp_root(),
        suffix=".csv",
        prefix="gauge_batch_",
        mode="w",
        newline="",
        encoding="utf-8-sig",
    )
    with temp_file as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "start", "end", "ratio", "reading"])
        for row in rows:
            writer.writerow([row["filename"], row["start"], row["end"], row["ratio"], row["reading"]])
    return temp_file.name


def create_zip_file(rows):
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, dir=get_runtime_temp_root(), suffix=".zip", prefix="gauge_batch_images_"
    )
    zip_path = temp_file.name
    temp_file.close()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for row in rows:
            image = row.get("download_image")
            if image is None:
                continue
            archive_name = f"{Path(row['filename']).stem or 'result'}_result.png"
            zip_file.writestr(archive_name, image_to_png_bytes(image))
    return zip_path


def resolve_point_mode(label):
    mapping = {"起始点": "start", "结束点": "end", "指针尖端": "pointer_tip", "指针根部": "pointer_root", "圆心点": "center"}
    return mapping.get(label, "none")


def index_html():
    index_path = Path(current_dir) / "index.html"
    return index_path.read_text(encoding="utf-8")


def list_image_paths(input_dir):
    resolved = (input_dir or "").strip()
    if not resolved:
        raise FileNotFoundError("请输入图片文件夹路径")
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"目录不存在: {resolved}")
    image_paths = sorted(
        [
            os.path.join(resolved, name)
            for name in os.listdir(resolved)
            if os.path.isfile(os.path.join(resolved, name)) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        ]
    )
    if not image_paths:
        raise FileNotFoundError(f"目录中未找到图片: {resolved}")
    return image_paths


def run_batch_job(job_id, input_dir, use_stn, use_yolo):
    try:
        image_paths = list_image_paths(input_dir)
        _batch_jobs[job_id]["progress"]["total"] = len(image_paths)

        with _infer_lock:
            logic = get_app_logic()
            batch_logic = BatchInferenceJSON(get_cfg())
            batch_logic.sync_runtime_from(logic)

            rows = []
            for index, image_path in enumerate(image_paths, start=1):
                logger.info("web batch inference processing file: %s", image_path)
                try:
                    with Image.open(image_path) as pil_image:
                        rgb_image = pil_image.convert("RGB")
                        _, reading, start_val, end_val = batch_logic.process_image(rgb_image, use_stn, use_yolo)
                    result_image = batch_logic.draw_visualization_with_value()
                    rows.append(
                        {
                            "filename": os.path.basename(image_path),
                            "thumbnail": image_to_data_url(result_image, max_width=220),
                            "full_image": image_to_data_url(result_image),
                            "start": format_metric(start_val),
                            "end": format_metric(end_val),
                            "ratio": format_metric(batch_logic.current_ratio),
                            "reading": format_metric(reading),
                            "download_image": result_image,
                        }
                    )
                except Exception as exc:
                    fallback = np.full((120, 180, 3), 245, dtype=np.uint8)
                    cv2.putText(fallback, "ERROR", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 50, 50), 2)
                    rows.append(
                        {
                            "filename": os.path.basename(image_path),
                            "thumbnail": image_to_data_url(fallback, max_width=220),
                            "full_image": image_to_data_url(fallback),
                            "start": "-",
                            "end": "-",
                            "ratio": "-",
                            "reading": f"推理失败: {exc}",
                            "download_image": fallback,
                        }
                    )
                _batch_jobs[job_id]["progress"]["completed"] = index

        _batch_jobs[job_id]["status"] = "packaging"
        _batch_jobs[job_id]["rows"] = [{k: v for k, v in row.items() if k != "download_image"} for row in rows]

        zip_path = create_zip_file(rows)
        csv_path = create_csv_file(rows)
        _batch_jobs[job_id]["status"] = "completed"
        _batch_jobs[job_id]["downloads"] = {
            "zip": register_download(zip_path, build_timestamped_filename("gauge_batch_images", "zip"), "application/zip"),
            "csv": register_download(csv_path, build_timestamped_filename("gauge_batch_results", "csv"), "text/csv"),
        }
    except Exception as exc:
        _batch_jobs[job_id]["status"] = "failed"
        _batch_jobs[job_id]["error"] = str(exc)


@app.get("/", response_class=HTMLResponse)
def home():
    return index_html()


@app.get("/styles.css")
def styles():
    return FileResponse(Path(current_dir) / "styles.css", media_type="text/css")


@app.get("/app.js")
def script():
    return FileResponse(Path(current_dir) / "app.js", media_type="application/javascript")


@app.get("/api/bootstrap")
def bootstrap():
    cfg = get_cfg()
    options = default_model_options()
    defaults = {
        "model_path": cfg.predict.model_path,
        "stn_path": cfg.predict.stn_model_path,
        "yolo_path": cfg.predict.yolo_model_path,
    }
    loaded = get_app_logic().textnet is not None
    return {**options, "defaults": defaults, "loaded": loaded, "instructions": INSTRUCTIONS}


@app.post("/api/models/load")
def load_models(payload: LoadModelsPayload):
    with _infer_lock:
        try:
            cfg, resolved_config_path = build_cfg_for_reader_model(
                model_path=payload.model_path, stn_path=payload.stn_path, yolo_path=payload.yolo_path
            )
            logic = reset_app_logic(cfg)
            logic.load_models(payload.model_path, payload.stn_path, payload.yolo_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "message": "模型加载完成",
        "config_path": resolved_config_path,
        "config_mode": "matched" if resolve_reader_config_path(payload.model_path) else "default",
    }


@app.post("/api/infer")
async def infer(image: UploadFile = File(...), use_stn: bool = Form(True), use_yolo: bool = Form(True)):
    with _infer_lock:
        logic = get_app_logic()
        if logic.textnet is None:
            raise HTTPException(status_code=400, detail="请先加载模型")

        raw_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"无法读取图片: {exc}") from exc

        result_image, reading, start_value, end_value = logic.process_image(pil_image, use_stn=use_stn, use_yolo=use_yolo)
        if result_image is None:
            raise HTTPException(status_code=400, detail=str(reading))

        payload = state_payload(result_image=result_image, reading=reading)
        payload["start_value"] = format_metric(start_value)
        payload["end_value"] = format_metric(end_value)
        return payload


@app.post("/api/session/update-point")
def update_point(payload: UpdatePointPayload):
    with _infer_lock:
        logic = get_app_logic()
        point_mode = resolve_point_mode(payload.mode)
        if point_mode == "none":
            raise HTTPException(status_code=400, detail="请选择有效的修正模式")
        result_image, reading = logic.update_point(point_mode, int(payload.x), int(payload.y))
        if isinstance(reading, str) and ("模型未加载" in reading or "请先运行推理" in reading):
            raise HTTPException(status_code=400, detail=reading)
        return state_payload(result_image=result_image, reading=reading)


@app.post("/api/session/update-value")
def update_value(payload: UpdateValuePayload):
    with _infer_lock:
        logic = get_app_logic()
        if payload.field == "start":
            result_image, reading = logic.update_start_val(payload.value)
        elif payload.field == "end":
            result_image, reading = logic.update_end_val(payload.value)
        else:
            raise HTTPException(status_code=400, detail="不支持的字段")

        if result_image is None:
            raise HTTPException(status_code=400, detail=str(reading))
        return state_payload(result_image=result_image, reading=reading)


@app.post("/api/batch/jobs")
def create_batch_job(payload: BatchJobPayload):
    logic = get_app_logic()
    if logic.textnet is None:
        raise HTTPException(status_code=400, detail="请先加载模型")

    try:
        image_paths = list_image_paths(payload.input_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    _batch_jobs[job_id] = {
        "status": "running",
        "rows": [],
        "downloads": {},
        "error": None,
        "progress": {"completed": 0, "total": len(image_paths)},
    }
    worker = threading.Thread(
        target=run_batch_job, args=(job_id, payload.input_dir, payload.use_stn, payload.use_yolo), daemon=True
    )
    worker.start()
    return {"job_id": job_id}


@app.get("/api/batch/jobs/{job_id}")
def get_batch_job(job_id: str):
    payload = _batch_jobs.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="批量任务不存在")
    return payload


@app.get("/api/download/{file_id}")
def download(file_id: str):
    payload = _download_cache.get(file_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="下载文件不存在或已过期")
    return FileResponse(payload["path"], media_type=payload["media_type"], filename=payload["filename"])


def run_server(host="127.0.0.1", port=11451, open_browser=True):
    cleanup_runtime_cache()
    get_cfg().print_config()
    web_url = f"http://{host}:{port}/"

    if open_browser:

        def open_browser_later():
            try:
                webbrowser.open(web_url)
            except Exception as exc:
                logger.warning("Failed to open browser automatically: %s", exc)

        threading.Timer(1.0, open_browser_later).start()

    uvicorn.run(app, host=host, port=port)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gauge Read Native Web")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web Host")
    parser.add_argument("--port", type=int, default=11451, help="Web Port")
    args = parser.parse_args(argv)

    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("Native web console log level set to DEBUG")

    run_server(host=args.host, port=args.port, open_browser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
