import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import webview
import multiprocessing
import warnings
import socket
from tkinter import messagebox

multiprocessing.set_start_method("spawn", force=True)
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
    messagebox.showerror("Error", f"Could not find a free port in the range {start_port}-{end_port}.")
    os._exit(1)


def launcher(server_name, server_port):
    # 此处实现gradio的启动
    import app as gradio_app

    print(f"WebUI launching on {server_name}:{server_port}")
    gradio_app.demo.launch(inbrowser=False, share=False, server_name=server_name, server_port=server_port)


def start_gradio(server_name, server_port):
    gradio_process = multiprocessing.Process(target=launcher, args=(server_name, server_port))
    gradio_process.start()
    return gradio_process


def main():
    server_name = "127.0.0.1"
    server_port = find_free_port(server_name)
    isdebug = False
    try:
        gradio_process = start_gradio(server_name, server_port)
        webview.create_window(
            title="Gauge Reader GUI",
            url=f"index.html?ip={server_name}&port={server_port}",
            width=1300,
            height=850,
            frameless=False,
            easy_drag=False,
            text_select=False,
            confirm_close=True,
        )
        webview.start(debug=isdebug, http_server=False)
        gradio_process.terminate()
        gradio_process.join()
    except Exception as e:
        import traceback

        messagebox.showerror("Error", f"Failed to start the webview: {e}\n{traceback.format_exc()}")
    os._exit(0)


if __name__ == "__main__":
    main()
