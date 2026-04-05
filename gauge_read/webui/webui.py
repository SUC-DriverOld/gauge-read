import gradio as gr
import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(current_dir)
repo_root = os.path.dirname(package_root)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from gauge_read.utils.config import AttrDict
from gauge_read.utils.logger import logger
from gauge_read.webui.webui_logic import GaugeAppWebUI


BATCH_RESULT_PLACEHOLDER = (
    "<div style='padding:12px;text-align:center;color:var(--body-text-color-subdued, var(--color-text-secondary, #57606a));'>"
    "点击“开始批量推理”按钮后开始批量处理"
    "</div>"
)
INSTRUCTIONS = """
    1. **加载模型**：在下拉菜单中选择预训练的仪表读数模型、STN矫正模型和YOLO检测模型，然后点击“加载模型”按钮。
    2. **单图推理**：上传一张仪表图片，选择是否启用STN矫正和YOLO检测，然后点击“开始推理”按钮。处理后的图片会显示在右侧，并且会自动识别起始值和结束值。
    3. **编辑结果**：如果识别结果不准确，可以选择修正模式（起始点、结束点、指针尖端、指针根部、圆心点），然后点击图片上的对应位置进行修正。也可以直接编辑起始值和结束值文本框来调整读数。
    4. **批量推理**：输入包含多张仪表图片的文件夹路径，选择是否启用STN矫正和YOLO检测，然后点击“开始批量推理”按钮。处理结果会以表格格式显示，并提供ZIP文件下载处理后的图片，以及CSV文件下载读数结果。
"""


def init_runtime(config_path=None):
    global cfg, app_logic
    resolved = config_path or os.environ.get("GAUGE_CONFIG", AttrDict.DEFAULT_CONFIG_PATH)
    cfg = AttrDict(resolved)
    app_logic = GaugeAppWebUI(cfg)
    logger.debug("WebUI runtime initialized with config=%s", resolved)


init_runtime()


def get_model_files(directory):
    if not os.path.exists(directory):
        logger.warning("Model directory does not exist: %s", directory)
        return []
    valid_ext = {".pt", ".pth"}
    files = []
    for f in os.listdir(directory):
        p = os.path.join(directory, f)
        if os.path.isfile(p) and os.path.splitext(f)[1].lower() in valid_ext:
            files.append(p)
    files = sorted(files)
    logger.debug("Discovered %s model files under %s", len(files), directory)
    return files


meter_dir = os.path.join(repo_root, "pretrain", "meter")
stn_dir = os.path.join(repo_root, "pretrain", "stn")
yolo_dir = os.path.join(repo_root, "pretrain", "yolo")

os.makedirs(meter_dir, exist_ok=True)
os.makedirs(stn_dir, exist_ok=True)
os.makedirs(yolo_dir, exist_ok=True)

model_options = get_model_files(meter_dir)
stn_options = get_model_files(stn_dir)
yolo_options = get_model_files(yolo_dir)


def load_models_ui(model_path, stn_path, yolo_path):
    logger.info("WebUI load_models triggered: textnet=%s, stn=%s, yolo=%s", model_path, stn_path, yolo_path)
    app_logic.load_models(model_path, stn_path, yolo_path)
    model_selection = gr.update(open=False)
    return model_path, stn_path, yolo_path, model_selection


def process_ui(image, use_stn, use_yolo):
    if image is None:
        logger.warning("WebUI single-image inference invoked without an uploaded image")
        return None, "请上传图片", 0.0, 0.0
    logger.info("WebUI single-image inference triggered: use_stn=%s, use_yolo=%s", use_stn, use_yolo)
    vis_img, val, start_val, end_val = app_logic.process_image(image, use_stn, use_yolo)
    return vis_img, val, start_val, end_val


def process_batch_ui(input_dir, use_stn, use_yolo, progress=None):
    if progress is None:
        progress = gr.Progress(track_tqdm=True)
    logger.info("WebUI batch inference triggered: input_dir=%s, use_stn=%s, use_yolo=%s", input_dir, use_stn, use_yolo)
    result_html, zip_path, csv_path = app_logic.process_batch_directory(input_dir, use_stn, use_yolo, progress=progress)
    logger.info("WebUI batch inference finished: zip=%s, csv=%s", zip_path, csv_path)
    return result_html, zip_path, csv_path


def update_point_ui(evt: gr.SelectData, mode):
    # evt.index is [x, y]
    x, y = evt.index[0], evt.index[1]

    type_map = {"起始点": "start", "结束点": "end", "指针尖端": "pointer_tip", "指针根部": "pointer_root", "圆心点": "center"}
    point_type = type_map.get(mode, "none")

    if point_type != "none":
        logger.debug("WebUI point update requested: mode=%s, x=%s, y=%s", point_type, x, y)
        new_img, new_val = app_logic.update_point(point_type, x, y)
        return new_img, new_val
    else:
        return gr.skip(), gr.skip()


def update_start_val_ui(text):
    logger.debug("WebUI start value updated manually: %s", text)
    new_img, new_val = app_logic.update_start_val(text)
    return new_img, new_val


def update_end_val_ui(text):
    logger.debug("WebUI end value updated manually: %s", text)
    new_img, new_val = app_logic.update_end_val(text)
    return new_img, new_val


with gr.Blocks(title="模拟仪表读数系统") as demo:
    gr.HTML("""<h1 style="text-align: center;">模拟仪表读数系统</h1>""")
    with gr.Accordion("使用说明 (点击展开)", open=False):
        gr.Markdown(INSTRUCTIONS)
    with gr.Accordion("模型选择", open=True) as model_selection:
        with gr.Group():
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=model_options, label="仪表读数模型", value=model_options[0] if model_options else None
                )
                stn_dropdown = gr.Dropdown(
                    choices=stn_options, label="STN矫正模型", value=stn_options[0] if stn_options else None
                )
                yolo_dropdown = gr.Dropdown(
                    choices=yolo_options, label="YOLO检测模型", value=yolo_options[0] if yolo_options else None
                )
            load_btn = gr.Button("加载模型", variant="primary")
    with gr.Tabs():
        with gr.TabItem("单图推理"):
            with gr.Group():
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="上传仪表图片", type="pil", height=500)
                        with gr.Row():
                            use_stn_chk = gr.Checkbox(label="启用STN矫正", value=True)
                            use_yolo_chk = gr.Checkbox(label="启用YOLO检测", value=True)
                        run_btn = gr.Button("开始推理", variant="primary")
                    with gr.Column():
                        edit_mode = gr.Radio(
                            choices=["未选择", "起始点", "结束点", "指针尖端", "指针根部", "圆心点"],
                            value="未选择",
                            label="修正模式 (选择要移动的点然后点击图片)",
                        )
                        result_image = gr.Image(label="处理后的图片 (点击编辑)", interactive=False, height=410)
                        with gr.Row():
                            start_val_input = gr.Textbox(label="起始值 (可手动编辑)", value="0", interactive=True)
                            end_val_input = gr.Textbox(label="结束值 (可手动编辑)", value="0", interactive=True)
                            result_val = gr.Textbox(label="读数结果", lines=1, scale=2)
        with gr.TabItem("批量推理"):
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=4):
                        batch_input_dir = gr.Textbox(
                            label="图片文件夹",
                            placeholder="请输入待批量推理的图片文件夹路径",
                            value=cfg.predict.get("data_dir", ""),
                        )
                    with gr.Column(scale=1):
                        batch_use_stn_chk = gr.Checkbox(label="启用STN矫正", value=True)
                        batch_use_yolo_chk = gr.Checkbox(label="启用YOLO检测", value=True)
                batch_run_btn = gr.Button("开始批量推理", variant="primary")
            with gr.Group():
                with gr.Row():
                    batch_zip_file = gr.File(label="下载结果图像 ZIP", interactive=False)
                    batch_csv_file = gr.File(label="下载结果 CSV", interactive=False)
                batch_result_html = gr.HTML(label="批量推理结果", value=BATCH_RESULT_PLACEHOLDER)

    load_btn.click(
        load_models_ui,
        inputs=[model_dropdown, stn_dropdown, yolo_dropdown],
        outputs=[model_dropdown, stn_dropdown, yolo_dropdown, model_selection],
    )
    run_btn.click(
        process_ui,
        inputs=[input_image, use_stn_chk, use_yolo_chk],
        outputs=[result_image, result_val, start_val_input, end_val_input],
    )
    batch_run_btn.click(
        process_batch_ui,
        inputs=[batch_input_dir, batch_use_stn_chk, batch_use_yolo_chk],
        outputs=[batch_result_html, batch_zip_file, batch_csv_file],
    )
    result_image.select(update_point_ui, inputs=[edit_mode], outputs=[result_image, result_val])
    start_val_input.change(update_start_val_ui, inputs=[start_val_input], outputs=[result_image, result_val])
    end_val_input.change(update_end_val_ui, inputs=[end_val_input], outputs=[result_image, result_val])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gauge Read WebUI")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("WebUI console log level set to DEBUG")

    logger.info("Starting Gauge Read WebUI with config=%s", args.config or "default")

    if args.config:
        init_runtime(args.config)

    cfg.print_config()

    demo.queue().launch(inbrowser=True)
