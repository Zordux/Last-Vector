from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import psutil
import uvicorn

from .run_store import RunStore


def create_app(runs_dir: Path) -> FastAPI:
    app = FastAPI(title="Last-Vector Training Dashboard")
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
    store = RunStore(runs_dir)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        runs = [store.run_summary(run_id) for run_id in store.list_runs()]
        return templates.TemplateResponse("index.html", {"request": request, "runs": runs})

    @app.get("/api/runs", response_class=JSONResponse)
    async def api_runs() -> JSONResponse:
        return JSONResponse([store.run_summary(run_id) for run_id in store.list_runs()])

    @app.get("/api/hw", response_class=JSONResponse)
    async def api_hw() -> JSONResponse:
        vm = psutil.virtual_memory()
        return JSONResponse(
            {
                "cpu_percent": psutil.cpu_percent(),
                "ram_percent": vm.percent,
                "ram_used_gb": round(vm.used / 1e9, 2),
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    app = create_app(Path(args.runs_dir))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
