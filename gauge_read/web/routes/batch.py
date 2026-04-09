import threading
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from gauge_read.web import core
from gauge_read.web.schemas import BatchJobPayload


router = APIRouter()


@router.post("/api/batch/uploads")
async def upload_batch_images(images: list[UploadFile] = File(...)):
    try:
        saved_dir, saved_count = core.save_uploaded_batch_images(images)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        for image in images:
            await image.close()

    return {"input_dir": saved_dir, "count": saved_count}


@router.post("/api/batch/jobs")
def create_batch_job(payload: BatchJobPayload):
    logic = core.get_app_logic()
    if logic.textnet is None:
        raise HTTPException(status_code=400, detail="请先加载模型")

    try:
        image_paths = core.list_image_paths(payload.input_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = uuid.uuid4().hex
    core.batch_jobs[job_id] = {
        "status": "running",
        "rows": [],
        "downloads": {},
        "error": None,
        "progress": {"completed": 0, "total": len(image_paths)},
    }
    worker = threading.Thread(
        target=core.run_batch_job, args=(job_id, payload.input_dir, payload.use_stn, payload.use_yolo), daemon=True
    )
    worker.start()
    return {"job_id": job_id}


@router.get("/api/batch/jobs/{job_id}")
def get_batch_job(job_id: str):
    payload = core.batch_jobs.get(job_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="批量任务不存在")
    return payload
