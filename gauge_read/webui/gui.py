import os
import socket
import warnings
import argparse
import webview
from tkinter import messagebox

from gauge_read.utils.logger import logger

gradio_app = None
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
warnings.filterwarnings("ignore")
index_path = os.path.join(os.path.dirname(__file__), "index.html")


def find_free_port(ip, start_port=11451, end_port=19198):
    logger.debug("Searching for a free port on %s in range [%s, %s]", ip, start_port, end_port)
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((ip, port))
                logger.info("Selected free port for desktop GUI: %s:%s", ip, port)
                return port
            except OSError:
                continue
    logger.error("No free port found in range [%s, %s] for ip=%s", start_port, end_port, ip)
    messagebox.showerror("错误", f"无法在端口范围 {start_port}-{end_port} 内找到可用端口！")
    os._exit(1)


def start_gradio(server_name, server_port, config_path=None):
    if config_path:
        os.environ["GAUGE_CONFIG"] = config_path
        logger.info("Desktop GUI set GAUGE_CONFIG=%s", config_path)

    from gauge_read.webui import webui

    webui.cfg.print_config()

    global gradio_app
    gradio_app = webui

    logger.info("Desktop GUI launching embedded WebUI on %s:%s", server_name, server_port)
    gradio_app.demo.launch(
        inbrowser=False, share=False, server_name=server_name, server_port=server_port, prevent_thread_lock=True
    )


def main():
    parser = argparse.ArgumentParser(description="Gauge Read Desktop GUI")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("WebUI console log level set to DEBUG")

    server_name = "127.0.0.1"
    server_port = find_free_port(server_name)
    logger.info("Starting desktop GUI with config=%s", args.config or "default")

    try:
        webview.create_window(
            title="模拟仪表读数系统",
            url=f"{index_path}?ip={server_name}&port={server_port}",
            width=1600,
            height=900,
            frameless=False,
            easy_drag=False,
            text_select=False,
            confirm_close=True,
        )
        webview.start(func=start_gradio, args=(server_name, server_port, args.config), debug=False, http_server=False)
        if gradio_app:
            logger.info("Closing embedded Gradio app from desktop GUI")
            gradio_app.demo.close()
    except Exception as e:
        import traceback

        logger.exception("Desktop GUI failed to start webview")
        messagebox.showerror("错误", f"启动 webview 失败: {e}\n{traceback.format_exc()}")
    os._exit(0)


if __name__ == "__main__":
    main()
