import os
import socket
import sys
import warnings
from tkinter import messagebox

import webview

current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(current_dir)
repo_root = os.path.dirname(package_root)
if repo_root not in sys.path:
    sys.path.append(repo_root)

gradio_app = None
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
warnings.filterwarnings("ignore")


def find_free_port(ip, start_port=11451, end_port=19198):
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((ip, port))
                return port
            except OSError:
                continue
    messagebox.showerror("错误", f"无法在端口范围 {start_port}-{end_port} 内找到可用端口！")
    os._exit(1)


def start_gradio(server_name, server_port):
    from gauge_read.webui import app

    global gradio_app
    gradio_app = app

    print(f"WebUI launching on {server_name}:{server_port}")
    gradio_app.demo.launch(
        inbrowser=False, share=False, server_name=server_name, server_port=server_port, prevent_thread_lock=True
    )


def main():
    server_name = "127.0.0.1"
    server_port = find_free_port(server_name)

    try:
        webview.create_window(
            title="模拟仪表读数系统",
            url=f"index.html?ip={server_name}&port={server_port}",
            width=1600,
            height=900,
            frameless=False,
            easy_drag=False,
            text_select=False,
            confirm_close=True,
        )
        webview.start(func=start_gradio, args=(server_name, server_port), debug=False, http_server=False)
        if gradio_app:
            gradio_app.demo.close()
    except Exception as e:
        import traceback

        messagebox.showerror("错误", f"启动 webview 失败: {e}\n{traceback.format_exc()}")
    os._exit(0)


if __name__ == "__main__":
    main()
