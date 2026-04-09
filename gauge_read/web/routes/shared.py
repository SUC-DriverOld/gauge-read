from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from gauge_read.web import core
from gauge_read.web.schemas import LoadModelsPayload


router = APIRouter()


@router.get("/api/bootstrap")
def bootstrap():
    cfg = core.get_cfg()
    options = core.default_model_options()
    defaults = {
        "model_path": cfg.predict.model_path,
        "stn_path": cfg.predict.stn_model_path,
        "yolo_path": cfg.predict.yolo_model_path,
    }
    return {**options, "defaults": defaults, "instructions": core.INSTRUCTIONS}


@router.post("/api/models/load")
def load_models(payload: LoadModelsPayload):
    with core.infer_lock:
        try:
            cfg, resolved_config_path = core.build_cfg_for_reader_model(
                model_path=payload.model_path, stn_path=payload.stn_path, yolo_path=payload.yolo_path
            )
            logic = core.reset_app_logic(cfg)
            logic.load_models(payload.model_path, payload.stn_path, payload.yolo_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "message": "模型加载完成",
        "config_path": resolved_config_path,
        "config_mode": "matched" if core.resolve_reader_config_path(payload.model_path) else "default",
    }


@router.get("/api/download/{file_id}")
def download(file_id: str):
    payload = core.download_cache.get(file_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="下载文件不存在或已过期")
    return FileResponse(payload["path"], media_type=payload["media_type"], filename=payload["filename"])
