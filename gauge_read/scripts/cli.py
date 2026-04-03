import os
import sys
import argparse
import json
import cv2

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(current_dir)
repo_root = os.path.dirname(package_root)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from gauge_read.webui.app_logic import GaugeAppModel
from gauge_read.utils.config import AttrDict
from gauge_read.utils.logger import logger


def emit_json(payload, exit_code=0):
    print(json.dumps(payload, indent=2))
    if exit_code:
        sys.exit(exit_code)


def emit_json_error(message, exit_code=1):
    emit_json({"status": "error", "message": message}, exit_code=exit_code)


def main():
    parser = argparse.ArgumentParser(description="Gauge Reader CLI")

    parser.add_argument("-c", "--config", type=str, default=None, help="Path to config YAML")

    # Inputs
    parser.add_argument("image_path", type=str, help="Path to the gauge image")

    # Model Paths
    parser.add_argument("--yolo", type=str, default=None, help="Path to YOLO weights")
    parser.add_argument("--stn", type=str, default=None, help="Path to STN weights")
    parser.add_argument("--textnet", type=str, default=None, help="Path to TextNet weights")

    # Flags
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLO detection")
    parser.add_argument("--use-stn", action="store_true", help="Enable STN correction")

    # Overrides
    parser.add_argument("--start-value", type=float, default=None, help="Override start scale value")
    parser.add_argument("--end-value", type=float, default=None, help="Override end scale value")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Path to save result image (optional)")

    args = parser.parse_args()

    logger.info("CLI invocation started")
    logger.debug(
        "CLI arguments parsed: image_path=%s, config=%s, use_yolo=%s, use_stn=%s, output=%s, start_override=%s, end_override=%s",
        args.image_path,
        args.config or AttrDict.DEFAULT_CONFIG_PATH,
        args.use_yolo,
        args.use_stn,
        args.output,
        args.start_value,
        args.end_value,
    )

    cfg = AttrDict(args.config or AttrDict.DEFAULT_CONFIG_PATH)
    logger.info("Configuration loaded for CLI: %s", args.config or AttrDict.DEFAULT_CONFIG_PATH)

    yolo_path = args.yolo or cfg.predict.get("yolo_model_path", "pretrain/best.pt")
    stn_path = args.stn if args.stn is not None else cfg.data.get("stn_model_path", "")
    textnet_path = args.textnet or cfg.predict.get("model_path", "")
    logger.info("Resolved model paths for CLI: textnet=%s, stn=%s, yolo=%s", textnet_path, stn_path or "disabled", yolo_path)

    # Validate Image
    if not os.path.exists(args.image_path):
        logger.error("CLI input image does not exist: %s", args.image_path)
        emit_json_error(f"Image not found: {args.image_path}")

    image = cv2.imread(args.image_path)
    if image is None:
        logger.error("CLI failed to read image structure from path: %s", args.image_path)
        emit_json_error("Failed to read image structure")

    logger.info("CLI input image loaded successfully: %s", args.image_path)
    logger.debug("CLI input image shape: %s", image.shape)

    # Initialize Model
    # Note: GaugeAppModel assumes 'cfg' which loads config.
    # Since we are in scripts/, ensure config works.
    # Usually config.py uses relative paths or is loaded once.

    try:
        logger.info("Initializing GaugeAppModel for CLI inference")
        app_logic = GaugeAppModel(cfg)
        app_logic.load_models(textnet_path=textnet_path, stn_path=stn_path, yolo_path=yolo_path)
    except Exception as e:
        logger.exception("CLI model initialization failed")
        emit_json_error(f"Model initialization failed: {str(e)}")

    # Process
    try:
        logger.info("Starting CLI inference: use_stn=%s, use_yolo=%s", args.use_stn, args.use_yolo)
        vis_img, val, start_val, end_val = app_logic.process_image(image, use_stn=args.use_stn, use_yolo=args.use_yolo)
        logger.debug("Initial CLI inference output: value=%s, start=%s, end=%s", val, start_val, end_val)

        if val is None:
            logger.error("CLI inference returned no value: %s", vis_img)
            emit_json_error(f"Inference failed: {vis_img}")

        # Overrides
        need_recalc = False
        if args.start_value is not None:
            app_logic.current_start_value = args.start_value
            need_recalc = True
            logger.info("Applied CLI start value override: %s", args.start_value)

        if args.end_value is not None:
            app_logic.current_end_value = args.end_value
            need_recalc = True
            logger.info("Applied CLI end value override: %s", args.end_value)

        final_val = val
        if need_recalc:
            logger.info("Recalculating CLI result after manual overrides")
            final_val = app_logic.recalculate()

        final_ratio = getattr(app_logic, "current_ratio", 0.0)
        logger.info(
            "CLI inference completed successfully: measure_value=%s, ratio=%s, start_value=%s, end_value=%s",
            final_val,
            final_ratio,
            app_logic.current_start_value,
            app_logic.current_end_value,
        )

        # Save output image if requested
        if args.output:
            # app_logic.draw_visualization() uses current state
            # update_point uses draw_visualization internally
            # process_image returns display_img (RGB from app_logic)
            # But we might have recalculated.
            # Let's force a redraw.
            vis_res = app_logic.draw_visualization()
            save_ok = cv2.imwrite(args.output, cv2.cvtColor(vis_res, cv2.COLOR_RGB2BGR))
            if save_ok:
                logger.info("CLI result image saved to %s", args.output)
            else:
                logger.error("CLI failed to save result image to %s", args.output)
                emit_json_error(f"Failed to save output image: {args.output}")

        # Result
        result = {
            "status": "success",
            "measure_value": final_val,
            "ratio": final_ratio,
            "start_value": app_logic.current_start_value,
            "end_value": app_logic.current_end_value,
        }

        emit_json(result)

    except Exception as e:
        logger.exception("CLI processing failed")
        emit_json_error(f"Processing error: {str(e)}")


if __name__ == "__main__":
    main()
