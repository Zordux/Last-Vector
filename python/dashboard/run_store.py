from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_runs(self) -> List[str]:
        return sorted([p.name for p in self.root.iterdir() if p.is_dir()])

    def run_summary(self, run_id: str) -> Dict[str, Any]:
        run_dir = self.root / run_id
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.jsonl"
        ckpt_dir = run_dir / "checkpoints"
        best_model = run_dir / "best_model.zip"

        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

        last_metric = {}
        if metrics_path.exists():
            lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
            if lines:
                last_metric = json.loads(lines[-1])

        return {
            "run_id": run_id,
            "config": config,
            "last_metric": last_metric,
            "checkpoints": sorted([p.name for p in ckpt_dir.glob("*.zip")]) if ckpt_dir.exists() else [],
            "best_model": str(best_model) if best_model.exists() else None,
            "state": "running" if last_metric else "stopped",
        }
