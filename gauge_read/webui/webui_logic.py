import gradio as gr

from gauge_read.utils.logger import logger
from gauge_read.utils.app_logic import GaugeApp
from gauge_read.webui.batch_infer import BatchInference


class GaugeAppWebUI(GaugeApp):
    """WebUI wrapper that keeps Gradio-specific behavior out of the core reader."""

    @staticmethod
    def notify_error(message):
        try:
            gr.Error(message)
        except Exception:
            logger.debug("Gradio error notification skipped: %s", message)

    @staticmethod
    def notify_info(message):
        try:
            gr.Info(message)
        except Exception:
            logger.debug("Gradio info notification skipped: %s", message)

    def process_batch_directory(self, input_dir, use_stn=True, use_yolo=False, progress=None):
        batch_inference = BatchInference(self.cfg)
        batch_inference.sync_runtime_from(self)
        return batch_inference.process_directory(input_dir, use_stn=use_stn, use_yolo=use_yolo, progress=progress)

    def update_point(self, point_type, x, y):
        if self.textnet is None or self.detector is None:
            return gr.skip(), "模型未加载"
        if self.current_image is None:
            return gr.skip(), "请先运行推理"

        x, y = int(x), int(y)
        if point_type == "start":
            if not self.current_std_points:
                self.current_std_points = [(0, 0), (0, 0)]
            self.current_std_points[0] = (x, y)
        elif point_type == "end":
            if len(self.current_std_points) < 2:
                self.current_std_points.append((0, 0))
            self.current_std_points[1] = (x, y)
        elif point_type == "pointer_tip":
            if not self.current_pointer_line:
                self.current_pointer_line = [(0, 0), (0, 0)]
            self.current_pointer_line[1] = (x, y)
        elif point_type == "pointer_root":
            if not self.current_pointer_line:
                self.current_pointer_line = [(0, 0), (0, 0)]
            self.current_pointer_line[0] = (x, y)
        elif point_type == "center":
            self.current_center = (x, y)

        return self.draw_visualization(), self.recalculate()

    def update_start_val(self, text):
        if self.textnet is None or self.detector is None:
            return gr.skip(), "模型未加载"
        if self.current_image is None:
            return gr.skip(), "请先运行推理"

        try:
            self.current_start_value = float(text)
        except ValueError:
            return gr.skip(), "起始值输入无效"
        return self.draw_visualization(), self.recalculate()

    def update_end_val(self, text):
        if self.textnet is None or self.detector is None:
            return gr.skip(), "模型未加载"
        if self.current_image is None:
            return gr.skip(), "请先运行推理"

        try:
            self.current_end_value = float(text)
        except ValueError:
            return gr.skip(), "结束值输入无效"
        return self.draw_visualization(), self.recalculate()
