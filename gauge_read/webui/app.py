import gradio as gr
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(current_dir)
repo_root = os.path.dirname(package_root)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from gauge_read.webui.app_logic import GaugeAppModel

app_logic = GaugeAppModel()


def get_model_files(directory):
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pth")]


ckpt_dir = os.path.join(repo_root, "pretrain")
log_dir = os.path.join(repo_root, "logs")
stn_dir = os.path.join(log_dir, "stn_convnext")

model_options = get_model_files(ckpt_dir) + get_model_files(os.path.join(log_dir, "multimodal_convnext"))
stn_options = get_model_files(stn_dir)


def load_models_ui(model_path, stn_path):
    app_logic.load_models(model_path, stn_path)
    return model_path, stn_path


def process_ui(image, use_stn, use_yolo):
    if image is None:
        return None, "请上传图片", 0.0, 0.0
    vis_img, val, start_val, end_val = app_logic.process_image(image, use_stn, use_yolo)
    return vis_img, val, start_val, end_val


def update_point_ui(evt: gr.SelectData, mode):
    # evt.index is [x, y]
    x, y = evt.index[0], evt.index[1]

    point_type = "none"
    if mode == "移动起始点":
        point_type = "start"
    elif mode == "移动结束点":
        point_type = "end"
    elif mode == "移动指针尖端":
        point_type = "pointer_tip"
    elif mode == "移动指针根部":
        point_type = "pointer_root"
    elif mode == "移动圆心点":
        point_type = "center"

    if point_type != "none":
        new_img, new_val = app_logic.update_point(point_type, x, y)
        return new_img, new_val
    else:
        return gr.skip(), gr.skip()


def update_start_val_ui(text):
    new_img, new_val = app_logic.update_start_val(text)
    return new_img, new_val


def update_end_val_ui(text):
    new_img, new_val = app_logic.update_end_val(text)
    return new_img, new_val


with gr.Blocks(title="模拟仪表读数系统") as demo:
    gr.HTML("""<h1 style="text-align: center;">模拟仪表读数系统</h1>""")
    with gr.Group():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        choices=model_options, label="仪表读数模型", value=model_options[0] if model_options else None
                    )
                    stn_dropdown = gr.Dropdown(
                        choices=stn_options, label="STN矫正模型", value=stn_options[0] if stn_options else None
                    )
                load_btn = gr.Button("加载模型", variant="primary")
                input_image = gr.Image(label="上传仪表图片", type="pil", height=500)
                with gr.Row():
                    use_stn_chk = gr.Checkbox(label="启用STN矫正", value=True)
                    use_yolo_chk = gr.Checkbox(label="启用YOLO检测", value=True)
                run_btn = gr.Button("开始推理", variant="primary")
            with gr.Column():
                edit_mode = gr.Radio(
                    choices=["未选择", "移动起始点", "移动结束点", "移动指针尖端", "移动指针根部", "移动圆心点"],
                    value="未选择",
                    label="修正模式 (选择要移动的点然后点击图片)",
                )
                result_image = gr.Image(label="处理后的图片 (点击编辑)", interactive=False, height=450)
                with gr.Row():
                    start_val_input = gr.Textbox(label="起始值", value="0", interactive=True)
                    end_val_input = gr.Textbox(label="结束值", value="0", interactive=True)
                result_val = gr.Textbox(label="读数结果", lines=1, scale=2)

    load_btn.click(load_models_ui, inputs=[model_dropdown, stn_dropdown], outputs=[model_dropdown, stn_dropdown])
    run_btn.click(
        process_ui,
        inputs=[input_image, use_stn_chk, use_yolo_chk],
        outputs=[result_image, result_val, start_val_input, end_val_input],
    )
    result_image.select(update_point_ui, inputs=[edit_mode], outputs=[result_image, result_val])
    start_val_input.change(update_start_val_ui, inputs=[start_val_input], outputs=[result_image, result_val])
    end_val_input.change(update_end_val_ui, inputs=[end_val_input], outputs=[result_image, result_val])

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
