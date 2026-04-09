import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from gauge_read.web import core


router = APIRouter()


@router.post("/api/realtime/frame")
async def realtime_frame(
    image: UploadFile = File(...),
    use_stn: bool = Form(True),
    use_yolo: bool = Form(True),
    manual_values: bool = Form(False),
    start_value: float = Form(0.0),
    end_value: float = Form(0.0),
):
    with core.infer_lock:
        logic = core.get_app_logic()
        if logic.textnet is None:
            raise HTTPException(status_code=400, detail="请先加载模型")

        raw_bytes = await image.read()
        try:
            pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"无法读取图片: {exc}") from exc

        runtime_logic = core.NativeGaugeApp(core.get_cfg())
        runtime_logic.sync_runtime_from(logic)

        try:
            result_image, reading, detected_start, detected_end = runtime_logic.process_image(
                pil_image, use_stn=use_stn, use_yolo=use_yolo
            )
        except Exception as exc:
            return {
                "valid": False,
                "reason": str(exc),
                "ocr_error": False,
                "start_value": core.format_metric(start_value),
                "end_value": core.format_metric(end_value),
            }

        if result_image is None:
            return {
                "valid": False,
                "reason": str(reading),
                "ocr_error": False,
                "start_value": core.format_metric(start_value),
                "end_value": core.format_metric(end_value),
            }

        if manual_values:
            runtime_logic.current_start_value = float(start_value)
            runtime_logic.current_end_value = float(end_value)
            runtime_logic.current_ocr_error = False
            runtime_logic.recalculate()

        ocr_failed = bool(runtime_logic.current_ocr_error)
        if ocr_failed:
            return {
                "valid": False,
                "reason": "OCR失败",
                "ocr_error": True,
                "start_value": core.format_metric(detected_start),
                "end_value": core.format_metric(detected_end),
            }

        return {
            "valid": True,
            "reading_label": "读数结果",
            "reading": core.format_metric(runtime_logic.current_value),
            "ratio": core.format_metric(runtime_logic.current_ratio),
            "ocr_error": False,
            "start_value": core.format_metric(runtime_logic.current_start_value if manual_values else detected_start),
            "end_value": core.format_metric(runtime_logic.current_end_value if manual_values else detected_end),
        }
