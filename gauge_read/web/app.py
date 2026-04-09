import argparse
import threading
import uvicorn
import webbrowser

from gauge_read.utils.logger import logger
from gauge_read.web import core
from gauge_read.web.server import create_app


app = create_app()


def run_server(host="127.0.0.1", port=8080, open_browser=True):
    core.cleanup_runtime_cache()
    core.get_cfg().print_config()
    web_url = f"http://{host}:{port}/"

    if open_browser:

        def open_browser_later():
            try:
                webbrowser.open(web_url)
            except Exception as exc:
                logger.warning("Failed to open browser automatically: %s", exc)

        threading.Timer(1.0, open_browser_later).start()

    uvicorn.run(app, host=host, port=port)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gauge Read Native Web")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web Host")
    parser.add_argument("--port", type=int, default=8080, help="Web Port")
    args = parser.parse_args(argv)

    if args.debug:
        import logging

        logger.console_handler.setLevel(logging.DEBUG)
        logger.info("Native web console log level set to DEBUG")

    run_server(host=args.host, port=args.port, open_browser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
