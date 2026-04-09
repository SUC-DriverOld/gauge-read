from typing import Optional

from pydantic import BaseModel


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
