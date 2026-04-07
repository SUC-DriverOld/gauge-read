import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gauge_read.utils.config import AttrDict
from gauge_read.utils.logger import logger
from gauge_read.utils.app_logic import GaugeApp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="Gauge validation script")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("-c", "--config", type=str, default=AttrDict.DEFAULT_CONFIG_PATH, help="Path to config YAML")
    parser.add_argument(
        "-i", "--input-dir", type=str, required=True, help="Validation directory containing images/ and labels/"
    )
    parser.add_argument("--use_yolo", action="store_true", help="Enable YOLO detection during validation")
    parser.add_argument("--use_stn", action="store_true", help="Enable STN correction during validation")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to export validation_summary.json and validation_details.csv"
    )
    return parser.parse_args()


def _is_close(lhs, rhs, atol=1e-6):
    return math.isclose(float(lhs), float(rhs), abs_tol=atol, rel_tol=0.0)


def _to_float(value, field_name, source_path):
    if value is None:
        raise ValueError(f"Missing field '{field_name}' in {source_path}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{field_name}' must be numeric in {source_path}, got {value!r}") from exc


def _load_label(label_path):
    with label_path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    if not isinstance(data, dict):
        raise ValueError(f"Label file must contain a JSON object: {label_path}")

    return {
        "filename": data.get("filename", label_path.with_suffix("").name),
        "start": _to_float(data.get("start"), "start", label_path),
        "end": _to_float(data.get("end"), "end", label_path),
        "full": _to_float(data.get("full"), "full", label_path),
        "value": _to_float(data.get("value"), "value", label_path),
    }


def _collect_pairs(input_dir):
    input_path = Path(input_dir)
    image_dir = input_path / "images"
    label_dir = input_path / "labels"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    image_map = {
        image_path.stem: image_path
        for image_path in sorted(image_dir.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    }
    label_map = {
        label_path.stem: label_path
        for label_path in sorted(label_dir.iterdir())
        if label_path.is_file() and label_path.suffix.lower() == ".json"
    }

    if not image_map:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if not label_map:
        raise FileNotFoundError(f"No labels found in {label_dir}")

    missing_labels = sorted(set(image_map) - set(label_map))
    missing_images = sorted(set(label_map) - set(image_map))
    for stem in missing_labels:
        logger.warning("Skipping image without label: %s", image_map[stem])
    for stem in missing_images:
        logger.warning("Skipping label without image: %s", label_map[stem])

    shared_stems = sorted(set(image_map) & set(label_map))
    if not shared_stems:
        raise RuntimeError(f"No matched image/label pairs found under {input_path}")

    return [(image_map[stem], label_map[stem]) for stem in shared_stems]


def _safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _safe_acc(values, threshold):
    if not values:
        return None
    return sum(1 for value in values if value <= threshold) / len(values)


def _make_metric_block(errors):
    return {
        "count": len(errors),
        "mean_error_rate": _safe_mean(errors),
        "acc@2%": _safe_acc(errors, 0.02),
        "acc@5%": _safe_acc(errors, 0.05),
        "acc@10%": _safe_acc(errors, 0.10),
    }


def _is_ocr_error(infer_value):
    return isinstance(infer_value, str) and infer_value.startswith("OCR error")


def _is_inference_failure(vis_image, infer_value):
    return vis_image is None or (isinstance(infer_value, str) and not _is_ocr_error(infer_value))


def _write_summary_json(summary, output_dir):
    output_path = Path(output_dir) / "validation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    return output_path


def _write_details_csv(rows, output_dir):
    output_path = Path(output_dir) / "validation_details.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "label",
        "status",
        "failure_reason",
        "ocr_success",
        "ocr_start_correct",
        "ocr_end_correct",
        "ocr_both_correct",
        "gt_start",
        "gt_end",
        "gt_full",
        "gt_value",
        "infer_start",
        "infer_end",
        "infer_value",
        "raw_ratio",
        "normalized_ratio",
        "gt_ratio",
        "ratio_error",
        "value_error",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return output_path


def _save_figure(fig, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_metric_overview(summary, charts_dir):
    labels = [
        "OCR Success",
        "OCR Acc Overall",
        "OCR Acc on Success",
        "Ratio Acc@2%",
        "Ratio Acc@5%",
        "Ratio Acc@10%",
        "Reading Acc@2%",
        "Reading Acc@5%",
        "Reading Acc@10%",
    ]
    values = [
        summary["ocr"]["success_rate"] or 0.0,
        summary["ocr"]["accuracy_overall"] or 0.0,
        summary["ocr"]["accuracy_on_successful_ocr"] or 0.0,
        summary["ratio"]["acc@2%"] or 0.0,
        summary["ratio"]["acc@5%"] or 0.0,
        summary["ratio"]["acc@10%"] or 0.0,
        summary["reading"]["acc@2%"] or 0.0,
        summary["reading"]["acc@5%"] or 0.0,
        summary["reading"]["acc@10%"] or 0.0,
    ]
    colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#F58518", "#F58518", "#F58518", "#54A24B", "#54A24B", "#54A24B"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Validation Metric Overview")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    return _save_figure(fig, charts_dir / "metric_overview.png")


def _plot_error_distribution(detail_rows, charts_dir):
    ratio_errors = [row["ratio_error"] for row in detail_rows if isinstance(row.get("ratio_error"), (int, float))]
    reading_errors = [row["value_error"] for row in detail_rows if isinstance(row.get("value_error"), (int, float))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if ratio_errors:
        axes[0].hist(ratio_errors, bins=20, color="#F58518", edgecolor="white", alpha=0.9)
    axes[0].set_title("Ratio Error Distribution")
    axes[0].set_xlabel("Normalized Error")
    axes[0].set_ylabel("Sample Count")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    if reading_errors:
        axes[1].hist(reading_errors, bins=20, color="#54A24B", edgecolor="white", alpha=0.9)
    axes[1].set_title("Reading Error Distribution")
    axes[1].set_xlabel("Normalized Error")
    axes[1].set_ylabel("Sample Count")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    return _save_figure(fig, charts_dir / "error_distribution.png")


def _plot_prediction_scatter(detail_rows, charts_dir):
    ratio_points = [
        (row["gt_ratio"], row["normalized_ratio"])
        for row in detail_rows
        if isinstance(row.get("gt_ratio"), (int, float)) and isinstance(row.get("normalized_ratio"), (int, float))
    ]
    reading_points = [
        (row["gt_value"], row["infer_value"])
        for row in detail_rows
        if isinstance(row.get("gt_value"), (int, float)) and isinstance(row.get("infer_value"), (int, float))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if ratio_points:
        gt_ratio, pred_ratio = zip(*ratio_points, strict=True)
        axes[0].scatter(gt_ratio, pred_ratio, alpha=0.7, s=24, color="#F58518")
        lower = min(min(gt_ratio), min(pred_ratio))
        upper = max(max(gt_ratio), max(pred_ratio))
        axes[0].plot([lower, upper], [lower, upper], linestyle="--", color="#333333", linewidth=1)
    axes[0].set_title("GT Ratio vs Predicted Ratio")
    axes[0].set_xlabel("GT Ratio")
    axes[0].set_ylabel("Predicted Ratio")
    axes[0].grid(linestyle="--", alpha=0.3)

    if reading_points:
        gt_value, pred_value = zip(*reading_points, strict=True)
        axes[1].scatter(gt_value, pred_value, alpha=0.7, s=24, color="#54A24B")
        lower = min(min(gt_value), min(pred_value))
        upper = max(max(gt_value), max(pred_value))
        axes[1].plot([lower, upper], [lower, upper], linestyle="--", color="#333333", linewidth=1)
    axes[1].set_title("GT Reading vs Predicted Reading")
    axes[1].set_xlabel("GT Reading")
    axes[1].set_ylabel("Predicted Reading")
    axes[1].grid(linestyle="--", alpha=0.3)

    return _save_figure(fig, charts_dir / "prediction_scatter.png")


def _plot_status_distribution(detail_rows, charts_dir):
    status_counter = Counter(row.get("status", "unknown") for row in detail_rows)
    labels = list(status_counter.keys())
    values = list(status_counter.values())

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title("Validation Sample Status Distribution")
    ax.set_xlabel("Status")
    ax.set_ylabel("Sample Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center", va="bottom", fontsize=9)
    return _save_figure(fig, charts_dir / "status_distribution.png")


def _plot_ocr_category_distribution(detail_rows, charts_dir):
    categories = Counter()
    for row in detail_rows:
        if row.get("ocr_success") is True:
            if row.get("ocr_both_correct") is True:
                categories["OCR Correct"] += 1
            else:
                categories["OCR Success but Wrong"] += 1
        elif row.get("ocr_success") is False:
            categories["OCR Failed"] += 1
        else:
            categories["Not Evaluated"] += 1

    labels = list(categories.keys())
    values = list(categories.values())
    colors = ["#54A24B", "#EECA3B", "#E45756", "#B279A2"][: len(labels)]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.pie(
        values, labels=labels, autopct="%.1f%%", startangle=90, colors=colors, wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )
    ax.set_title("OCR Outcome Distribution")
    return _save_figure(fig, charts_dir / "ocr_distribution.png")


def _generate_charts(summary, detail_rows, output_dir):
    charts_dir = Path(output_dir)
    chart_paths = {
        "metric_overview": str(_plot_metric_overview(summary, charts_dir).resolve()),
        "error_distribution": str(_plot_error_distribution(detail_rows, charts_dir).resolve()),
        "prediction_scatter": str(_plot_prediction_scatter(detail_rows, charts_dir).resolve()),
        "status_distribution": str(_plot_status_distribution(detail_rows, charts_dir).resolve()),
        "ocr_distribution": str(_plot_ocr_category_distribution(detail_rows, charts_dir).resolve()),
    }
    logger.info("Validation charts exported to %s", charts_dir)
    return chart_paths


def run_validation(cfg, input_dir, use_yolo=False, use_stn=False, config_path=None, output_dir=None):
    pairs = _collect_pairs(input_dir)
    logger.info(
        "Validation started: input_dir=%s, samples=%s, use_yolo=%s, use_stn=%s", input_dir, len(pairs), use_yolo, use_stn
    )
    if output_dir is None:
        output_dir = input_dir
        logger.warning("No output directory specified, using input directory for exports: %s", output_dir)
    else:
        logger.info("Validation artifacts will be exported to: %s", output_dir)

    app_model = GaugeApp(cfg)
    app_model.load_models(
        textnet_path=cfg.predict.model_path, stn_path=cfg.predict.stn_model_path, yolo_path=cfg.predict.yolo_model_path
    )

    total_count = len(pairs)
    ocr_success_count = 0
    ocr_accurate_count = 0
    ratio_errors = []
    value_errors = []
    ratio_skipped_count = 0
    value_skipped_count = 0
    inference_failures = []
    detail_rows = []

    for index, (image_path, label_path) in enumerate(pairs, start=1):
        logger.info("Validating sample [%s/%s]: %s", index, total_count, image_path)
        detail = {
            "image": str(image_path),
            "label": str(label_path),
            "status": "pending",
            "failure_reason": "",
            "ocr_success": None,
            "ocr_start_correct": None,
            "ocr_end_correct": None,
            "ocr_both_correct": None,
            "gt_start": None,
            "gt_end": None,
            "gt_full": None,
            "gt_value": None,
            "infer_start": None,
            "infer_end": None,
            "infer_value": None,
            "raw_ratio": None,
            "normalized_ratio": None,
            "gt_ratio": None,
            "ratio_error": None,
            "value_error": None,
        }
        try:
            gt = _load_label(label_path)
            detail["gt_start"] = gt["start"]
            detail["gt_end"] = gt["end"]
            detail["gt_full"] = gt["full"]
            detail["gt_value"] = gt["value"]

            app_model.current_ratio = 0.0
            app_model.current_start_value = 0.0
            app_model.current_end_value = 0.0

            with Image.open(image_path) as pil_image:
                rgb_image = pil_image.convert("RGB")
                vis_image, infer_value, infer_start, infer_end = app_model.process_image(
                    rgb_image, use_stn=use_stn, use_yolo=use_yolo
                )
        except Exception as exc:
            logger.exception("Validation sample crashed: %s", image_path)
            inference_failures.append({"image": str(image_path), "reason": str(exc)})
            detail["status"] = "crashed"
            detail["failure_reason"] = str(exc)
            detail_rows.append(detail)
            ratio_skipped_count += 1
            value_skipped_count += 1
            continue

        detail["infer_start"] = infer_start
        detail["infer_end"] = infer_end
        detail["infer_value"] = infer_value
        detail["raw_ratio"] = app_model.current_ratio

        if _is_inference_failure(vis_image, infer_value):
            message = infer_value if isinstance(infer_value, str) else "unknown inference failure"
            logger.warning("Inference failed for %s: %s", image_path, message)
            inference_failures.append({"image": str(image_path), "reason": message})
            detail["status"] = "failed"
            detail["failure_reason"] = message
            detail_rows.append(detail)
            ratio_skipped_count += 1
            value_skipped_count += 1
            continue

        ocr_success = not _is_ocr_error(infer_value)
        detail["ocr_success"] = ocr_success
        if ocr_success:
            ocr_success_count += 1
            start_correct = _is_close(infer_start, gt["start"])
            end_correct = _is_close(infer_end, gt["end"])
            both_correct = start_correct and end_correct
            detail["ocr_start_correct"] = start_correct
            detail["ocr_end_correct"] = end_correct
            detail["ocr_both_correct"] = both_correct
            if both_correct:
                ocr_accurate_count += 1
        else:
            detail["ocr_start_correct"] = False
            detail["ocr_end_correct"] = False
            detail["ocr_both_correct"] = False

        full_span = gt["full"] - gt["start"]
        first_tick_span = gt["end"] - gt["start"]
        gt_ratio = None

        if abs(full_span) < EPS:
            logger.warning("Skipping ratio/value metrics due to zero full span: %s", label_path)
            detail["status"] = "skipped_zero_full_span"
            detail_rows.append(detail)
            ratio_skipped_count += 1
            if ocr_success:
                value_skipped_count += 1
            continue

        gt_ratio = (gt["value"] - gt["start"]) / full_span
        detail["gt_ratio"] = gt_ratio

        if abs(first_tick_span) < EPS:
            logger.warning("Skipping normalized ratio due to zero first-tick span: %s", label_path)
            detail["status"] = "skipped_zero_first_tick_span"
            ratio_skipped_count += 1
        else:
            normalization_factor = first_tick_span / full_span
            # current_ratio is relative to the start->first-major-tick interval,
            # so convert it to full-scale ratio by multiplying the interval fraction.
            infer_ratio = float(app_model.current_ratio) * normalization_factor
            ratio_error = abs(infer_ratio - gt_ratio)
            ratio_errors.append(ratio_error)
            detail["normalized_ratio"] = infer_ratio
            detail["ratio_error"] = ratio_error
            logger.debug(
                "Ratio metrics for %s: raw_ratio=%s normalized_ratio=%s gt_ratio=%s error=%s",
                image_path.name,
                app_model.current_ratio,
                infer_ratio,
                gt_ratio,
                ratio_error,
            )

        if not ocr_success:
            detail["status"] = "ocr_failed"
            detail_rows.append(detail)
            value_skipped_count += 1
            continue

        infer_value_float = float(infer_value)
        # Compare predicted reading against labeled ground-truth reading.
        value_error = abs(infer_value_float - gt["value"]) / full_span
        value_errors.append(value_error)
        detail["value_error"] = value_error
        if detail["status"] in {"pending", "skipped_zero_first_tick_span"}:
            detail["status"] = "ok"
        detail_rows.append(detail)
        logger.debug(
            "Value metrics for %s: infer_value=%s gt_value=%s normalized_error=%s",
            image_path.name,
            infer_value_float,
            gt["value"],
            value_error,
        )

    summary = {
        "config": str(Path(config_path or AttrDict.DEFAULT_CONFIG_PATH).resolve()),
        "input_dir": str(Path(input_dir).resolve()),
        "use_yolo": use_yolo,
        "use_stn": use_stn,
        "total_samples": total_count,
        "ocr": {
            "success_count": ocr_success_count,
            "success_rate": ocr_success_count / total_count if total_count else None,
            "accurate_count": ocr_accurate_count,
            "accuracy_overall": ocr_accurate_count / total_count if total_count else None,
            "accuracy_on_successful_ocr": ocr_accurate_count / ocr_success_count if ocr_success_count else None,
        },
        "ratio": {**_make_metric_block(ratio_errors), "skipped_count": ratio_skipped_count},
        "reading": {**_make_metric_block(value_errors), "skipped_count": value_skipped_count},
        "failures": {"count": len(inference_failures), "samples": inference_failures},
    }

    if output_dir:
        details_path = _write_details_csv(detail_rows, output_dir)
        chart_paths = _generate_charts(summary, detail_rows, output_dir)
        summary["exports"] = {
            "output_dir": str(Path(output_dir).resolve()),
            "details_csv": str(details_path.resolve()),
            "charts": chart_paths,
        }
        summary_path = _write_summary_json(summary, output_dir)
        summary["exports"]["summary_json"] = str(summary_path.resolve())
        logger.info("Validation artifacts exported: summary=%s details=%s", summary_path, details_path)

    logger.info(
        "Validation finished: total=%s, ocr_success=%s, ocr_accurate=%s, ratio_eval=%s, value_eval=%s, failures=%s",
        total_count,
        ocr_success_count,
        ocr_accurate_count,
        len(ratio_errors),
        len(value_errors),
        len(inference_failures),
    )
    return summary


def main():
    args = parse_args()
    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("Validation console log level set to DEBUG")
    cfg = AttrDict(args.config)
    summary = run_validation(
        cfg, args.input_dir, use_yolo=args.use_yolo, use_stn=args.use_stn, config_path=args.config, output_dir=args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
