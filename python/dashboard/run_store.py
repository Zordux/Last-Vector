from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_runs(self) -> List[str]:
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()], reverse=True)

    def _read_last_metric(self, metrics_path: Path) -> Dict[str, Any]:
        if not metrics_path.exists():
            return {}

        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return {}
        return rows[-1]

    def _latest_checkpoint(self, ckpt_dir: Path) -> Optional[str]:
        if not ckpt_dir.exists():
            return None
        checkpoints = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        return checkpoints[-1].name if checkpoints else None

    def run_summary(self, run_id: str) -> Dict[str, Any]:
        run_dir = self.root / run_id
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.csv"
        ckpt_dir = run_dir / "checkpoints"
        best_model = run_dir / "best_model.zip"
        status_path = run_dir / "status.json"
        tensorboard_dir = run_dir / "tensorboard"

        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        last_metric = self._read_last_metric(metrics_path)
        checkpoints = sorted([p.name for p in ckpt_dir.glob("*.zip")]) if ckpt_dir.exists() else []

        state = "stopped"
        status = {}
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            state = str(status.get("state", state))
        elif last_metric:
            state = "active"

        return {
            "run_id": run_id,
            "config": config,
            "last_metric": last_metric,
            "checkpoints": checkpoints,
            "current_checkpoint": self._latest_checkpoint(ckpt_dir),
            "best_model": str(best_model) if best_model.exists() else None,
            "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir.exists() else None,
            "status": status,
            "state": state,
        }
