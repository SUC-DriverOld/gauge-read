import base64
import csv
import html
import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gauge_read.utils.logger import logger
from gauge_read.utils.app_logic import GaugeApp


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_VISIBLE_BATCH_ROWS = 5
THEME_TEXT_COLOR = "var(--body-text-color, var(--color-text-primary, #1f2328))"
THEME_MUTED_TEXT_COLOR = "var(--body-text-color-subdued, var(--color-text-secondary, #57606a))"
THEME_BG_COLOR = "var(--block-background-fill, var(--background-fill-primary, #ffffff))"
THEME_SUBTLE_BG_COLOR = "var(--background-fill-secondary, rgba(127, 127, 127, 0.08))"
THEME_BORDER_COLOR = "var(--border-color-primary, var(--block-border-color, #d8dee4))"
THEME_ERROR_COLOR = "var(--error-text-color, #d1242f)"


class BatchInference(GaugeApp):
    def __init__(self, cfg):
        super().__init__(cfg)

    @staticmethod
    def _format_result_value(value):
        if isinstance(value, (int, float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    @staticmethod
    def _image_to_base64(image, max_width=220):
        if image is None:
            return ""

        image_np = np.asarray(image)
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        pil_image = Image.fromarray(image_np.astype(np.uint8))
        if pil_image.width > max_width:
            scale = max_width / float(pil_image.width)
            target_size = (max_width, max(1, int(pil_image.height * scale)))
            pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_result_table(self, rows):
        if not rows:
            return f"<div style='padding:12px;color:{THEME_TEXT_COLOR};background:{THEME_BG_COLOR};'>未找到可推理图片</div>"

        table_rows = []
        for row in rows:
            image_html = (
                f"<div style='display:flex;flex-direction:column;align-items:center;gap:8px;'>"
                f"<img src='data:image/png;base64,{row['image_b64']}' style='max-width:220px;border-radius:8px;border:1px solid {THEME_BORDER_COLOR};' />"
                f"<div style='font-size:12px;color:{THEME_MUTED_TEXT_COLOR};word-break:break-all;'>{html.escape(row['filename'])}</div>"
                f"</div>"
            )
            table_rows.append(
                "<tr>"
                f"<td style='padding:12px;border:1px solid {THEME_BORDER_COLOR};vertical-align:middle;text-align:center;background:{THEME_BG_COLOR};color:{THEME_TEXT_COLOR};'>{image_html}</td>"
                f"<td style='padding:12px;border:1px solid {THEME_BORDER_COLOR};text-align:center;background:{THEME_BG_COLOR};color:{THEME_TEXT_COLOR};'>{html.escape(row['start'])}</td>"
                f"<td style='padding:12px;border:1px solid {THEME_BORDER_COLOR};text-align:center;background:{THEME_BG_COLOR};color:{THEME_TEXT_COLOR};'>{html.escape(row['end'])}</td>"
                f"<td style='padding:12px;border:1px solid {THEME_BORDER_COLOR};text-align:center;background:{THEME_BG_COLOR};color:{THEME_TEXT_COLOR};'>{html.escape(row['ratio'])}</td>"
                f"<td style='padding:12px;border:1px solid {THEME_BORDER_COLOR};text-align:center;background:{THEME_BG_COLOR};color:{THEME_TEXT_COLOR};'>{html.escape(row['reading'])}</td>"
                "</tr>"
            )

        row_height_px = 170
        visible_rows = min(len(rows), MAX_VISIBLE_BATCH_ROWS)
        summary = (
            f"<div style='padding:8px 4px 12px 4px;color:{THEME_MUTED_TEXT_COLOR};font-size:13px;'>"
            f"共 {len(rows)} 条结果，可滚动查看其余结果。"
            f"</div>"
        )

        return (
            summary
            + f"<div style='overflow-x:auto;overflow-y:auto;max-height:{visible_rows * row_height_px}px;color:{THEME_TEXT_COLOR};'>"
            f"<table style='width:100%;border-collapse:collapse;background:{THEME_BG_COLOR};color:{THEME_TEXT_COLOR};'>"
            "<thead>"
            f"<tr style='background:{THEME_SUBTLE_BG_COLOR};position:sticky;top:0;z-index:1;'>"
            f"<th style='padding:12px;border:1px solid {THEME_BORDER_COLOR};background:{THEME_SUBTLE_BG_COLOR};color:{THEME_TEXT_COLOR};text-align:center;'>结果图片</th>"
            f"<th style='padding:12px;border:1px solid {THEME_BORDER_COLOR};background:{THEME_SUBTLE_BG_COLOR};color:{THEME_TEXT_COLOR};text-align:center;'>起始值</th>"
            f"<th style='padding:12px;border:1px solid {THEME_BORDER_COLOR};background:{THEME_SUBTLE_BG_COLOR};color:{THEME_TEXT_COLOR};text-align:center;'>结束值</th>"
            f"<th style='padding:12px;border:1px solid {THEME_BORDER_COLOR};background:{THEME_SUBTLE_BG_COLOR};color:{THEME_TEXT_COLOR};text-align:center;'>读数Ratio</th>"
            f"<th style='padding:12px;border:1px solid {THEME_BORDER_COLOR};background:{THEME_SUBTLE_BG_COLOR};color:{THEME_TEXT_COLOR};text-align:center;'>读数值</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{''.join(table_rows)}</tbody>"
            "</table>"
            "</div>"
        )

    @staticmethod
    def _sanitize_output_name(filename):
        return Path(filename).stem or "result"

    def _create_csv_file(self, rows):
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".csv", prefix="gauge_batch_", mode="w", newline="", encoding="utf-8-sig"
        )
        with temp_file as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["filename", "start", "end", "ratio", "reading"])
            for row in rows:
                writer.writerow([row["filename"], row["start"], row["end"], row["ratio"], row["reading"]])
        return temp_file.name

    def _create_zip_file(self, rows):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".zip", prefix="gauge_batch_images_")
        zip_path = temp_file.name
        temp_file.close()

        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for row in rows:
                image = row.get("download_image")
                if image is None:
                    continue
                image_np = np.asarray(image)
                if image_np.ndim == 2:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
                pil_image = Image.fromarray(image_np.astype(np.uint8))
                buffer = io.BytesIO()
                pil_image.save(buffer, format="PNG")
                archive_name = f"{self._sanitize_output_name(row['filename'])}_result.png"
                zip_file.writestr(archive_name, buffer.getvalue())

        return zip_path

    def _annotate_result_image(self, image, ratio, reading):
        if image is None:
            return None

        annotated = np.asarray(image).copy()
        if annotated.ndim == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2RGB)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (8, 8), (260, 76), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

        text_color = (255, 255, 255)
        cv2.putText(
            annotated,
            f"Radio: {self._format_result_value(ratio)}",
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Result: {self._format_result_value(reading)}",
            (18, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2,
            cv2.LINE_AA,
        )
        return annotated

    def process_directory(self, input_dir, use_stn=True, use_yolo=False, progress=None):
        if self.textnet is None:
            return f"<div style='padding:12px;color:{THEME_ERROR_COLOR};'>模型未加载</div>", None, None

        if progress is not None:
            progress(0, desc="正在检查输入目录")

        input_dir = (input_dir or "").strip()
        if not input_dir:
            return f"<div style='padding:12px;color:{THEME_ERROR_COLOR};'>请输入图片文件夹路径</div>", None, None
        if not os.path.isdir(input_dir):
            return (
                f"<div style='padding:12px;color:{THEME_ERROR_COLOR};'>目录不存在: {html.escape(input_dir)}</div>",
                None,
                None,
            )

        image_paths = sorted(
            [
                os.path.join(input_dir, name)
                for name in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, name)) and os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
            ]
        )
        if not image_paths:
            return (
                f"<div style='padding:12px;color:{THEME_ERROR_COLOR};'>目录中未找到图片: {html.escape(input_dir)}</div>",
                None,
                None,
            )

        rows = []
        total_images = len(image_paths)
        iterable = image_paths
        if progress is not None:
            iterable = progress.tqdm(iterable, desc="正在批量推理", total=total_images, unit="img")

        for image_path in iterable:
            logger.info("batch inference processing file: %s", image_path)
            try:
                with Image.open(image_path) as pil_image:
                    rgb_image = pil_image.convert("RGB")
                    vis_img, reading, start_val, end_val = self.process_image(rgb_image, use_stn, use_yolo)
                ratio = self.current_ratio
                result_image = self._annotate_result_image(
                    vis_img if vis_img is not None else np.array(rgb_image), ratio=ratio, reading=reading
                )
                rows.append(
                    {
                        "filename": os.path.basename(image_path),
                        "image_b64": self._image_to_base64(result_image),
                        "start": self._format_result_value(start_val),
                        "end": self._format_result_value(end_val),
                        "ratio": self._format_result_value(ratio),
                        "reading": self._format_result_value(reading),
                        "download_image": result_image,
                    }
                )
            except Exception as exc:
                fallback = np.full((120, 180, 3), 245, dtype=np.uint8)
                cv2.putText(fallback, "ERROR", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (210, 50, 50), 2)
                rows.append(
                    {
                        "filename": os.path.basename(image_path),
                        "image_b64": self._image_to_base64(fallback),
                        "start": "-",
                        "end": "-",
                        "ratio": "-",
                        "reading": f"推理失败: {exc}",
                        "download_image": fallback,
                    }
                )

        if progress is not None:
            progress(1, desc=f"批量推理完成，共 {total_images} 张")

        result_table = self._build_result_table(rows)
        zip_path = self._create_zip_file(rows)
        csv_path = self._create_csv_file(rows)
        return result_table, zip_path, csv_path
