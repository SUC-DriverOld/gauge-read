import argparse
import os
import socket
import threading
import time
from tkinter import messagebox

import webview

from gauge_read.utils.logger import logger
from gauge_read.web.app import run_server


main_window = None


def find_free_port(ip, start_port=11451, end_port=19198):
    logger.debug("Searching for a free port on %s in range [%s, %s]", ip, start_port, end_port)
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((ip, port))
                logger.info("Selected free port for desktop GUI: %s:%s", ip, port)
                return port
            except OSError:
                continue
    logger.error("No free port found in range [%s, %s] for ip=%s", start_port, end_port, ip)
    messagebox.showerror("错误", f"无法在端口范围 {start_port}-{end_port} 内找到可用端口！")
    os._exit(1)


def wait_for_server(host, port, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            try:
                sock.connect((host, port))
                return True
            except OSError:
                time.sleep(0.2)
    return False


def start_fastapi(server_name, server_port, config_path=None):
    if config_path:
        os.environ["GAUGE_CONFIG"] = config_path
        logger.info("Desktop GUI set GAUGE_CONFIG=%s", config_path)

    server_thread = threading.Thread(
        target=run_server,
        kwargs={"host": server_name, "port": server_port, "open_browser": False},
        daemon=True,
    )
    server_thread.start()

    if not wait_for_server(server_name, server_port):
        logger.error("Desktop GUI timed out waiting for FastAPI web on %s:%s", server_name, server_port)
        messagebox.showerror("错误", f"本地 Web 服务启动超时：{server_name}:{server_port}")
        os._exit(1)

    if main_window is not None:
        logger.info("Desktop GUI loading FastAPI web app at %s:%s", server_name, server_port)
        main_window.load_url(f"http://{server_name}:{server_port}/")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gauge Read Desktop GUI")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("Desktop GUI console log level set to DEBUG")

    server_name = "127.0.0.1"
    server_port = find_free_port(server_name)
    logger.info("Starting desktop GUI with config=%s", args.config or "default")

    global main_window
    try:
        main_window = webview.create_window(
            title="模拟仪表读数系统",
            url="about:blank",
            width=1600,
            height=900,
            frameless=False,
            easy_drag=False,
            text_select=False,
            confirm_close=True,
        )
        webview.start(
            func=start_fastapi,
            args=(server_name, server_port, args.config),
            debug=False,
            http_server=False,
        )
    except Exception as exc:
        import traceback

        logger.exception("Desktop GUI failed to start webview")
        messagebox.showerror("错误", f"启动 webview 失败: {exc}\n{traceback.format_exc()}")
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
