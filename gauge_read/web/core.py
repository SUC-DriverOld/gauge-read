import base64
import csv
import io
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gauge_read.utils.app_logic import GaugeApp
from gauge_read.utils.config import AttrDict
from gauge_read.utils.logger import logger


current_dir = Path(__file__).resolve().parent
package_root = current_dir.parent
repo_root = package_root.parent

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INSTRUCTIONS = [
    "先选择仪表读数模型、STN 矫正模型和 YOLO 检测模型，再点击加载模型。",
    "单图推理支持上传图片、启用或关闭 STN 与 YOLO，并展示识别结果与可编辑点位。",
    "如果结果不准，可以选择修正模式后点击图片，也可以直接修改起始值与结束值。",
    "批量推理支持上传多张图片，自动输出结果预览、CSV 与结果图片 ZIP 下载。",
]

infer_lock = threading.Lock()
_cfg: AttrDict | None = None
_app_logic: "NativeGaugeApp | None" = None
download_cache: dict[str, dict[str, str]] = {}
batch_jobs: dict[str, dict] = {}


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


def cleanup_runtime_cache():
    cache_root = repo_root / ".cache" / "web_runtime"
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)


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


def reset_app_logic(cfg):
    global _cfg, _app_logic
    _cfg = cfg
    _app_logic = NativeGaugeApp(cfg)
    return _app_logic


def get_model_files(directory):
    if not os.path.exists(directory):
        return []

    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in {".pt", ".pth"}:
            files.append(path)
    return sorted(files)


def default_model_options():
    meter_dir = repo_root / "pretrain" / "meter"
    stn_dir = repo_root / "pretrain" / "stn"
    yolo_dir = repo_root / "pretrain" / "yolo"
    meter_dir.mkdir(parents=True, exist_ok=True)
    stn_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)
    return {
        "model_options": get_model_files(str(meter_dir)),
        "stn_options": get_model_files(str(stn_dir)),
        "yolo_options": get_model_files(str(yolo_dir)),
    }


def resolve_reader_config_path(model_path):
    if not model_path:
        return None

    candidate = Path(model_path).with_suffix(".yaml")
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
    download_cache[file_id] = {"path": file_path, "filename": filename, "media_type": media_type}
    return f"/api/download/{file_id}"


def get_runtime_temp_root():
    runtime_root = repo_root / ".cache" / "web_runtime"
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


def save_uploaded_batch_images(files):
    if not files:
        raise ValueError("请至少选择一张图片")

    upload_root = get_runtime_temp_root() / "batch_uploads" / uuid.uuid4().hex
    upload_root.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    used_names = set()
    for index, file in enumerate(files, start=1):
        original_name = Path(file.filename or f"image_{index}.png").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            continue

        target_name = original_name
        stem = Path(original_name).stem or f"image_{index}"
        serial = 1
        while target_name in used_names:
            target_name = f"{stem}_{serial}{suffix}"
            serial += 1

        content = file.file.read()
        if not content:
            continue

        (upload_root / target_name).write_bytes(content)
        used_names.add(target_name)
        saved_count += 1

    if saved_count == 0:
        shutil.rmtree(upload_root, ignore_errors=True)
        raise ValueError("未检测到可用图片文件")

    return str(upload_root), saved_count


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
    return (current_dir / "index.html").read_text(encoding="utf-8")


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
        batch_jobs[job_id]["progress"]["total"] = len(image_paths)

        with infer_lock:
            logic = get_app_logic()
            batch_logic = NativeGaugeApp(get_cfg())
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
                batch_jobs[job_id]["progress"]["completed"] = index

        batch_jobs[job_id]["status"] = "packaging"
        batch_jobs[job_id]["rows"] = [{k: v for k, v in row.items() if k != "download_image"} for row in rows]

        zip_path = create_zip_file(rows)
        csv_path = create_csv_file(rows)
        batch_jobs[job_id]["status"] = "completed"
        batch_jobs[job_id]["downloads"] = {
            "zip": register_download(zip_path, build_timestamped_filename("gauge_batch_images", "zip"), "application/zip"),
            "csv": register_download(csv_path, build_timestamped_filename("gauge_batch_results", "csv"), "text/csv"),
        }
    except Exception as exc:
        batch_jobs[job_id]["status"] = "failed"
        batch_jobs[job_id]["error"] = str(exc)
