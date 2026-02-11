from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_runs(self) -> List[str]:
        return sorted((path.name for path in self.root.iterdir() if path.is_dir()), reverse=True)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"_error": f"Failed to parse {path.name}"}

    @staticmethod
    def _tail_lines(path: Path, limit: int = 50) -> List[str]:
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-limit:]

    def _read_metrics(self, metrics_path: Path) -> List[Dict[str, Any]]:
        if not metrics_path.exists():
            return []
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            return []
        return rows

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _list_checkpoints(self, ckpt_dir: Path) -> List[str]:
        if not ckpt_dir.exists():
            return []
        checkpoints: List[tuple[float, str]] = []
        for path in ckpt_dir.glob("*.zip"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            checkpoints.append((mtime, path.name))
        checkpoints.sort(key=lambda item: item[0], reverse=True)
        return [name for _, name in checkpoints]

    def run_summary(self, run_id: str) -> Dict[str, Any]:
        run_dir = self.root / run_id
        config = self._read_json(run_dir / "config.json")
        status = self._read_json(run_dir / "status.json")
        metrics_rows = self._read_metrics(run_dir / "metrics.csv")
        last_metric = metrics_rows[-1] if metrics_rows else {}

        checkpoints = self._list_checkpoints(run_dir / "checkpoints")
        best_model = run_dir / "best_model.zip"
        tensorboard_dir = run_dir / "tensorboard"

        state = "stopped"
        if metrics_rows:
            state = "active"
        if "state" in status:
            state = str(status.get("state", state))

        last_update_ts = self._to_float(last_metric.get("ts"), 0.0)
        last_update = "n/a"
        if last_update_ts > 0:
            last_update = datetime.fromtimestamp(last_update_ts, tz=timezone.utc).isoformat()

        error_log_lines = self._tail_lines(run_dir / "error.log", limit=50)
        train_log_lines = self._tail_lines(run_dir / "train.log", limit=50)

        return {
            "run_id": run_id,
            "state": state,
            "config": config,
            "status": status,
            "metrics": {
                "steps": self._to_int(last_metric.get("timesteps"), 0),
                "episodes": self._to_int(status.get("episodes"), 0),
                "best_reward": self._to_float(status.get("best_reward"), 0.0),
                "avg_reward": self._to_float(last_metric.get("ep_rew_mean"), 0.0),
                "avg_survival": self._to_float(last_metric.get("ep_len_mean"), 0.0),
                "kills": self._to_int(status.get("kills"), 0),
                "damage_taken": self._to_float(status.get("damage_taken"), 0.0),
                "fps": self._to_float(last_metric.get("fps"), 0.0),
                "last_update": last_update,
            },
            "checkpoints": checkpoints,
            "current_checkpoint": checkpoints[0] if checkpoints else None,
            "best_model": str(best_model) if best_model.exists() else None,
            "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir.exists() else None,
            "tensorboard_cmd": f"tensorboard --logdir {tensorboard_dir}" if tensorboard_dir.exists() else None,
            "train_log_lines": train_log_lines,
            "error_log_lines": error_log_lines,
        }
