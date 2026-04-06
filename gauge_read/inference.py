import os
import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Gauge Reader CLI")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("input_path", type=str, help="Path to a gauge image or a directory containing images")
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLO detection")
    parser.add_argument("--use-stn", action="store_true", help="Enable STN correction")
    parser.add_argument("--start-value", type=float, default=None, help="Override start scale value")
    parser.add_argument("--end-value", type=float, default=None, help="Override end scale value")
    parser.add_argument("--output", type=str, default=None, help="Path to save result image (optional)")
    args = parser.parse_args()

    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("CLI console log level set to DEBUG")

    logger.info("CLI invocation started")
    logger.debug(
        "CLI arguments parsed: input_path=%s, config=%s, use_yolo=%s, use_stn=%s, output=%s, start_override=%s, end_override=%s",
        args.input_path,
        args.config or AttrDict.DEFAULT_CONFIG_PATH,
        args.use_yolo,
        args.use_stn,
        args.output,
        args.start_value,
        args.end_value,
    )

    cfg = AttrDict(args.config or AttrDict.DEFAULT_CONFIG_PATH)
    logger.info("Configuration loaded for CLI: %s", args.config or AttrDict.DEFAULT_CONFIG_PATH)

    image_paths = collect_input_images(args.input_path)
    logger.info("Resolved %s input image(s) from %s", len(image_paths), args.input_path)

    textnet_path = cfg.predict.model_path
    stn_path = cfg.predict.stn_model_path
    yolo_path = cfg.predict.yolo_model_path
    logger.info(
        "Resolved model paths from config: textnet=%s, stn=%s, yolo=%s", textnet_path, stn_path or "disabled", yolo_path
    )

    logger.info("Initializing GaugeApp for CLI inference")
    app_logic = GaugeApp(cfg)
    app_logic.load_models(textnet_path=textnet_path, stn_path=stn_path, yolo_path=yolo_path)

    logger.info("Starting CLI inference: use_stn=%s, use_yolo=%s", args.use_stn, args.use_yolo)
    multiple = len(image_paths) > 1 or os.path.isdir(args.input_path)
    results = []
    for image_path in image_paths:
        output_path = build_output_path(args.output, image_path, multiple)
        results.append(
            process_single_image(
                app_logic,
                image_path,
                use_stn=args.use_stn,
                use_yolo=args.use_yolo,
                start_value=args.start_value,
                end_value=args.end_value,
                output_path=output_path,
            )
        )

    final_payload = results if multiple else results[0]
    json_output_path = build_json_output_path(args.output, args.input_path, multiple)
    if json_output_path:
        write_json_output(final_payload, json_output_path)
        logger.info("CLI result json saved to %s", json_output_path)


if __name__ == "__main__":
    main()
