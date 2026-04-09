import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from gauge_read.web import core
from gauge_read.web.schemas import UpdatePointPayload, UpdateValuePayload


router = APIRouter()


@router.post("/api/infer")
async def infer(image: UploadFile = File(...), use_stn: bool = Form(True), use_yolo: bool = Form(True)):
    with core.infer_lock:
        logic = core.get_app_logic()
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

        payload = core.state_payload(result_image=result_image, reading=reading)
        payload["start_value"] = core.format_metric(start_value)
        payload["end_value"] = core.format_metric(end_value)
        return payload


@router.post("/api/session/update-point")
def update_point(payload: UpdatePointPayload):
    with core.infer_lock:
        logic = core.get_app_logic()
        point_mode = core.resolve_point_mode(payload.mode)
        if point_mode == "none":
            raise HTTPException(status_code=400, detail="请选择有效的修正模式")
        result_image, reading = logic.update_point(point_mode, int(payload.x), int(payload.y))
        if isinstance(reading, str) and ("模型未加载" in reading or "请先运行推理" in reading):
            raise HTTPException(status_code=400, detail=reading)
        return core.state_payload(result_image=result_image, reading=reading)


@router.post("/api/session/update-value")
def update_value(payload: UpdateValuePayload):
    with core.infer_lock:
        logic = core.get_app_logic()
        if payload.field == "start":
            result_image, reading = logic.update_start_val(payload.value)
        elif payload.field == "end":
            result_image, reading = logic.update_end_val(payload.value)
        else:
            raise HTTPException(status_code=400, detail="不支持的字段")

        if result_image is None:
            raise HTTPException(status_code=400, detail=str(reading))
        return core.state_payload(result_image=result_image, reading=reading)
