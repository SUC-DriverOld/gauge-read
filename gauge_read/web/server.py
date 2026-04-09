from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from gauge_read.web import core
from gauge_read.web.routes.batch import router as batch_router
from gauge_read.web.routes.realtime import router as realtime_router
from gauge_read.web.routes.shared import router as shared_router
from gauge_read.web.routes.single import router as single_router


def create_app():
    app = FastAPI(title="Gauge Read Web")
    app.mount("/static", StaticFiles(directory=core.current_dir / "static"), name="static")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return core.index_html()

    app.include_router(shared_router)
    app.include_router(single_router)
    app.include_router(batch_router)
    app.include_router(realtime_router)
    return app
