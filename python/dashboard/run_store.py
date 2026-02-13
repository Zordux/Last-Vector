from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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
            with path.open("rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(max(0, file_size - 32768), 0)
                chunk = f.read().decode("utf-8", errors="replace")
                lines = chunk.splitlines()
        except OSError:
            return []
        return lines[-limit:]

    def _read_metrics(self, metrics_path: Path) -> List[Dict[str, Any]]:
        if not metrics_path.exists():
            return []
        try:
            with metrics_path.open("rb") as f:
                # Read header first
                header_line = f.readline().decode("utf-8", errors="replace")
                # Get file size and seek to tail
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(max(0, file_size - 32768), 0)
                chunk = f.read().decode("utf-8", errors="replace")
                lines = chunk.splitlines()
                # Take last 400 lines from chunk
                tail_lines = lines[-400:]
                # Prepend header so csv.DictReader works
                csv_content = header_line + "\n".join(tail_lines)
                rows = list(csv.DictReader(csv_content.splitlines()))
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

    @staticmethod
    def _fmt_ts(value: Any) -> str:
        ts = RunStore._to_float(value, 0.0)
        if ts <= 0.0:
            return "n/a"
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    @staticmethod
    def _tb_url(run_id: str) -> str:
        return f"http://localhost:6006/#scalars&regexInput={run_id}"

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

    def _extract_series(self, metrics_rows: List[Dict[str, Any]], limit: int = 300) -> Dict[str, List[float]]:
        rows = metrics_rows[-limit:]
        steps = [self._to_int(row.get("timesteps"), 0) for row in rows]
        rewards = [self._to_float(row.get("ep_rew_mean"), 0.0) for row in rows]
        ep_lens = [self._to_float(row.get("ep_len_mean"), 0.0) for row in rows]
        kills = [self._to_float(row.get("kills", row.get("last_ep_kills", 0.0)), 0.0) for row in rows]
        recent_rows = [
            {"step": steps[idx], "reward": rewards[idx], "episode_length": ep_lens[idx], "kills": kills[idx]}
            for idx in range(max(0, len(steps) - 20), len(steps))
        ]
        return {
            "steps": steps,
            "reward": rewards,
            "episode_length": ep_lens,
            "kills": kills,
            "recent_rows": recent_rows,
        }

    def run_summary(self, run_id: str) -> Dict[str, Any]:
        run_dir = self.root / run_id
        config = self._read_json(run_dir / "config.json")
        status = self._read_json(run_dir / "status.json")
        metrics_rows = self._read_metrics(run_dir / "metrics.csv")
        last_metric = metrics_rows[-1] if metrics_rows else {}

        checkpoints = self._list_checkpoints(run_dir / "checkpoints")
        best_model = run_dir / "best_model.zip"
        tensorboard_dir = run_dir / "tensorboard"

        state = str(status.get("state", "stopped" if not metrics_rows else "active"))

        total_steps = self._to_int(status.get("total_steps", config.get("total_steps")), 0)
        steps_done = self._to_int(status.get("steps_done", last_metric.get("timesteps")), 0)
        progress_pct = self._to_float(status.get("progress_pct"), 0.0)
        if total_steps > 0 and progress_pct <= 0.0:
            progress_pct = min(100.0, 100.0 * float(steps_done) / float(total_steps))

        fps = self._to_float(status.get("fps", last_metric.get("fps")), 0.0)
        episodes_done = self._to_int(status.get("episodes_done", last_metric.get("episodes_done")), 0)
        latest_reward = self._to_float(status.get("latest_reward", last_metric.get("last_ep_reward")), 0.0)
        latest_ep_len = self._to_float(status.get("latest_ep_len", last_metric.get("last_ep_len")), 0.0)
        last_update = self._fmt_ts(status.get("last_update_time", last_metric.get("ts")))

        best_score = self._to_float(status.get("best_score"), 0.0)
        best_model_path = status.get("best_model_path") or (str(best_model) if best_model.exists() else "n/a")

        error_log_lines = self._tail_lines(run_dir / "error.log", limit=50)
        train_log_lines = self._tail_lines(run_dir / "train.log", limit=50)

        return {
            "run_id": run_id,
            "state": state,
            "config": config,
            "status": status,
            "metrics": {
                "steps": steps_done,
                "total_steps": total_steps,
                "progress_pct": progress_pct,
                "episodes": episodes_done,
                "best_reward": best_score,
                "avg_reward": self._to_float(last_metric.get("ep_rew_mean"), latest_reward),
                "avg_survival": self._to_float(last_metric.get("ep_len_mean"), latest_ep_len),
                "latest_reward": latest_reward,
                "latest_ep_len": latest_ep_len,
                "kills": self._to_float(last_metric.get("kills", last_metric.get("last_ep_kills", 0.0)), 0.0),
                "damage_taken": self._to_float(last_metric.get("damage_taken"), 0.0),
                "fps": fps,
                "last_update": last_update,
            },
            "series": self._extract_series(metrics_rows),
            "checkpoints": checkpoints,
            "current_checkpoint": checkpoints[0] if checkpoints else None,
            "best_model": str(best_model) if best_model.exists() else None,
            "best_model_info": {
                "path": str(best_model_path),
                "score": best_score,
            },
            "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir.exists() else None,
            "tensorboard_cmd": f"tensorboard --logdir {tensorboard_dir}" if tensorboard_dir.exists() else None,
            "tensorboard_link": self._tb_url(run_id),
            "train_log_lines": train_log_lines,
            "error_log_lines": error_log_lines,
        }
