from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import psutil
import uvicorn

from .run_store import RunStore


def create_app(runs_dir: Path) -> FastAPI:
    app = FastAPI(title="Last-Vector Training Dashboard")
    template_root = Path(__file__).parent / "templates"
    static_root = Path(__file__).parent / "static"
    templates = Jinja2Templates(directory=str(template_root))
    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
    store = RunStore(runs_dir)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        run_ids = store.list_runs()
        runs = [store.run_summary(run_id) for run_id in run_ids]
        selected_run = runs[0] if runs else None
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "runs": runs,
                "selected_run": selected_run,
                "run_count": len(runs),
            },
        )

    @app.get("/api/runs", response_class=JSONResponse)
    async def api_runs() -> JSONResponse:
        return JSONResponse([store.run_summary(run_id) for run_id in store.list_runs()])

    @app.get("/api/hw", response_class=JSONResponse)
    async def api_hw() -> JSONResponse:
        virtual_memory = psutil.virtual_memory()
        return JSONResponse(
            {
                "cpu_percent": psutil.cpu_percent(interval=0.0),
                "ram_percent": virtual_memory.percent,
                "ram_used_gb": round(virtual_memory.used / 1e9, 2),
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Last-Vector dashboard over LAN.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    app = create_app(Path(args.runs_dir))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
