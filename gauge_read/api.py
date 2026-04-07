import os
import argparse
import threading
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from gauge_read.utils.app_logic import GaugeApp
from gauge_read.utils.config import AttrDict
from gauge_read.utils.tools import (
    build_json_output_path,
    build_output_path,
    collect_input_images,
    process_single_image,
    write_json_output,
)
from gauge_read.utils.logger import logger

app = FastAPI(title="Gauge Reader API")
_infer_lock = threading.Lock()
_app_logic: Optional[GaugeApp] = None
_cfg: Optional[AttrDict] = None
_output_path: Optional[str] = None


def _resolve_model_paths(cfg):
    yolo = cfg.predict.yolo_model_path
    stn = cfg.predict.stn_model_path
    textnet = cfg.predict.model_path
    return yolo, stn, textnet


def _get_cfg():
    global _cfg
    if _cfg is None:
        _cfg = AttrDict(AttrDict.DEFAULT_CONFIG_PATH)
        logger.info("API default configuration loaded: %s", AttrDict.DEFAULT_CONFIG_PATH)
    return _cfg


def init_app_logic():
    global _app_logic
    cfg = _get_cfg()
    yolo, stn, textnet = _resolve_model_paths(cfg)

    logger.info("Initializing Gauge Model for API: textnet=%s, stn=%s, yolo=%s", textnet, stn or "disabled", yolo)
    _app_logic = GaugeApp(cfg)
    _app_logic.load_models(textnet_path=textnet, stn_path=stn, yolo_path=yolo)
    logger.info("Gauge API model initialization completed")


class GaugeRequest(BaseModel):
    input_path: Optional[str] = None
    use_yolo: bool = False
    use_stn: bool = False
    start_value: Optional[float] = None
    end_value: Optional[float] = None


@app.post("/infer")
def infer(req: GaugeRequest):
    if _app_logic is None:
        logger.error("API infer called before model initialization")
        raise HTTPException(status_code=500, detail="Model is not initialized")

    input_path = req.input_path
    if not input_path:
        raise HTTPException(status_code=400, detail="Request must provide input_path or image_path")

    logger.info(
        "API infer request received: input_path=%s, use_yolo=%s, use_stn=%s, start_override=%s, end_override=%s",
        input_path,
        req.use_yolo,
        req.use_stn,
        req.start_value,
        req.end_value,
    )

    with _infer_lock:
        logger.debug("API inference lock acquired")
        try:
            image_paths = collect_input_images(input_path)
        except Exception as exc:
            logger.warning("API request input path invalid: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        multiple = len(image_paths) > 1 or os.path.isdir(input_path)
        results = []
        for image_path in image_paths:
            output_path = build_output_path(_output_path, image_path, multiple)
            results.append(
                process_single_image(
                    _app_logic,
                    image_path,
                    use_stn=req.use_stn,
                    use_yolo=req.use_yolo,
                    start_value=req.start_value,
                    end_value=req.end_value,
                    output_path=output_path,
                )
            )

        final_payload = results if multiple else results[0]
        json_output_path = build_json_output_path(_output_path, input_path, multiple)
        if json_output_path:
            write_json_output(final_payload, json_output_path)
            logger.info("API result json saved to %s", json_output_path)

        return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauge Reader API")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Directory or file prefix used to save result image/json"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API Host")
    parser.add_argument("--port", type=int, default=1145, help="API Port")
    args = parser.parse_args()

    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("API console log level set to DEBUG")

    _output_path = args.output
    _cfg = AttrDict(args.config or AttrDict.DEFAULT_CONFIG_PATH)
    logger.info(
        "Starting Gauge Reader API: host=%s, port=%s, config=%s, output=%s",
        args.host,
        args.port,
        args.config or AttrDict.DEFAULT_CONFIG_PATH,
        args.output,
    )
    init_app_logic()
    uvicorn.run(app, host=args.host, port=args.port)
