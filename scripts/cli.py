import os
import sys
import argparse
import json
import cv2

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from webui.gauge_logic import GaugeAppModel


def main():
    parser = argparse.ArgumentParser(description="Gauge Reader CLI")

    # Inputs
    parser.add_argument("image_path", type=str, help="Path to the gauge image")

    # Model Paths
    parser.add_argument("--yolo", type=str, default="pretrain/best.pt", help="Path to YOLO weights")
    parser.add_argument("--stn", type=str, default="logs/stn/stn_ep50_loss0.0108.pth", help="Path to STN weights")
    parser.add_argument("--textnet", type=str, default="pretrain/textgraph_convnext_tiny_100.pth", help="Path to TextNet weights")

    # Flags
    parser.add_argument("--use-yolo", action="store_true", help="Enable YOLO detection")
    parser.add_argument("--use-stn", action="store_true", help="Enable STN correction")

    # Overrides
    parser.add_argument("--start-value", type=float, default=None, help="Override start scale value")
    parser.add_argument("--end-value", type=float, default=None, help="Override end scale value")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Path to save result image (optional)")

    args = parser.parse_args()

    # Validate Image
    if not os.path.exists(args.image_path):
        print(json.dumps({"status": "error", "message": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    image = cv2.imread(args.image_path)
    if image is None:
        print(json.dumps({"status": "error", "message": "Failed to read image structure"}))
        sys.exit(1)

    # Initialize Model
    # Note: GaugeAppModel assumes 'cfg' which loads config.
    # Since we are in scripts/, ensure config works.
    # Usually config.py uses relative paths or is loaded once.

    try:
        app_logic = GaugeAppModel()
        load_res = app_logic.load_models(textnet_path=args.textnet, stn_path=args.stn, yolo_path=args.yolo)
    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Model initialization failed: {str(e)}"}))
        sys.exit(1)

    # Process
    try:
        vis_img, val, start_val, end_val = app_logic.process_image(image, use_stn=args.use_stn, use_yolo=args.use_yolo)

        if val is None:
            print(json.dumps({"status": "error", "message": f"Inference failed: {vis_img}"}))
            sys.exit(1)

        # Overrides
        need_recalc = False
        if args.start_value is not None:
            app_logic.current_start_value = args.start_value
            need_recalc = True

        if args.end_value is not None:
            app_logic.current_end_value = args.end_value
            need_recalc = True

        final_val = val
        if need_recalc:
            final_val = app_logic.recalculate()

        final_ratio = getattr(app_logic, "current_ratio", 0.0)

        # Save output image if requested
        if args.output:
            # app_logic.draw_visualization() uses current state
            # update_point uses draw_visualization internally
            # process_image returns display_img (RGB from app_logic)
            # But we might have recalculated.
            # Let's force a redraw.
            vis_res = app_logic.draw_visualization()
            cv2.imwrite(args.output, cv2.cvtColor(vis_res, cv2.COLOR_RGB2BGR))

        # Result
        result = {
            "status": "success",
            "measure_value": final_val,
            "ratio": final_ratio,
            "start_value": app_logic.current_start_value,
            "end_value": app_logic.current_end_value,
        }

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(json.dumps({"status": "error", "message": f"Processing error: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
