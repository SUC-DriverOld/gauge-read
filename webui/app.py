import gradio as gr
import os
import sys

# Ensure correct path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from webui.gauge_logic import GaugeAppModel

# Initialize Logic Model
app_logic = GaugeAppModel()


def get_model_files(directory):
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pth")]


# Check directories
ckpt_dir = os.path.join(parent_dir, "pretrain")
log_dir = os.path.join(parent_dir, "logs")
stn_dir = os.path.join(parent_dir, "logs", "stn")

model_options = get_model_files(ckpt_dir) + get_model_files(os.path.join(parent_dir, "logs", "meter_data"))
stn_options = get_model_files(stn_dir)

# --- Gradio Interface Functions ---


def load_models_ui(model_path, stn_path):
    # Removing use_stn from load, always trying to load if path present
    return app_logic.load_models(model_path, stn_path)


def process_ui(image, use_stn, use_yolo):
    if image is None:
        return None, "Please upload an image.", 0.0, 0.0
    vis_img, val, start_val, end_val = app_logic.process_image(image, use_stn, use_yolo)
    return vis_img, val, start_val, end_val


def update_point_ui(evt: gr.SelectData, mode):
    # evt.index is [x, y]
    x, y = evt.index[0], evt.index[1]

    point_type = "none"
    if mode == "Move Start Point (0)":
        point_type = "start"
    elif mode == "Move End Point (Max)":
        point_type = "end"
    elif mode == "Move Pointer Tip":
        point_type = "pointer_tip"
    elif mode == "Move Pointer Root":
        point_type = "pointer_root"

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


# --- Layout ---

with gr.Blocks(title="Analog Gauge Reader WebUI") as demo:
    gr.Markdown("# 🎛️ Analog Gauge Reader WebUI")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configuration")
            model_dropdown = gr.Dropdown(
                choices=model_options, label="Reading Model Path", value=model_options[0] if model_options else None
            )
            stn_dropdown = gr.Dropdown(
                choices=stn_options, label="STN Model Path", value=stn_options[0] if stn_options else None
            )
            # stn_check moved to inference
            load_btn = gr.Button("Load Models", variant="primary")
            load_msg = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Input")
            input_image = gr.Image(label="Upload Gauge Image", type="pil")
            with gr.Row():
                use_stn_chk = gr.Checkbox(label="Enable STN Correction", value=True)
                use_yolo_chk = gr.Checkbox(label="Enable YOLO Detection", value=False)
            run_btn = gr.Button("Run Inference", variant="primary")

    gr.Markdown("---")
    gr.Markdown("### 3. Interactive Result & Correction")

    with gr.Row():
        with gr.Column():
            # Interaction Mode
            edit_mode = gr.Radio(
                choices=["None", "Move Start Point (0)", "Move End Point (Max)", "Move Pointer Tip", "Move Pointer Root"],
                value="None",
                label="🖱️ Click Mode (Select what to move then click on image)",
            )
            # Result Image
            result_image = gr.Image(label="Processed Image (Click to Edit)", interactive=False)

        with gr.Column():
            result_val = gr.Textbox(label="Calculated Reading", lines=1, scale=2)
            with gr.Row():
                start_val_input = gr.Textbox(label="Start Value", value="0.0", interactive=True)
                end_val_input = gr.Textbox(label="End Value", value="0.0", interactive=True)

            gr.Markdown("""
            **How to adjust:**
            1. Select **Click Mode** on the left.
            2. Click on the **Processed Image** to move the corresponding point.
            3. The reading updates automatically.
            4. Edit **Start/End Values** to correct the scale range.
            """)

    # Events
    load_btn.click(load_models_ui, inputs=[model_dropdown, stn_dropdown], outputs=[load_msg])

    run_btn.click(
        process_ui,
        inputs=[input_image, use_stn_chk, use_yolo_chk],
        outputs=[result_image, result_val, start_val_input, end_val_input],
    )

    # Image Click Event for Editing
    result_image.select(update_point_ui, inputs=[edit_mode], outputs=[result_image, result_val])

    # Value Change Events
    start_val_input.change(update_start_val_ui, inputs=[start_val_input], outputs=[result_image, result_val])
    end_val_input.change(update_end_val_ui, inputs=[end_val_input], outputs=[result_image, result_val])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
