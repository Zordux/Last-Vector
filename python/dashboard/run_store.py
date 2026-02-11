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

        last_metric: Dict[str, Any] = {}
        if metrics_path.exists():
            lines = [ln for ln in metrics_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                last_metric = json.loads(lines[-1])

        checkpoints = sorted([p.name for p in ckpt_dir.glob("*.zip")]) if ckpt_dir.exists() else []
        current_ckpt = checkpoints[-1] if checkpoints else None

        return {
            "run_id": run_id,
            "config": config,
            "steps": int(last_metric.get("timesteps", 0)),
            "episodes": int(last_metric.get("episodes", 0)),
            "last_metric": last_metric,
            "checkpoints": checkpoints,
            "current_checkpoint": current_ckpt,
            "best_model": best_model.name if best_model.exists() else None,
            "tensorboard_dir": str(run_dir / "tensorboard"),
            "tensorboard_hint": f"tensorboard --logdir {run_dir / 'tensorboard'} --bind_all",
            "state": "running" if last_metric else "stopped",
        }
