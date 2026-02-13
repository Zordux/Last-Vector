from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

import last_vector_core


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve PPO inference actions over TCP.")
    parser.add_argument("--model", required=True, help="Path to SB3 PPO model .zip")
    parser.add_argument("--host", default="127.0.0.1", help="Listen host")
    parser.add_argument("--port", type=int, default=5555, help="Listen port")
    return parser.parse_args()


def json_dumps_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")


def recv_json_line(conn: socket.socket, recv_buffer: bytearray) -> dict[str, Any] | None:
    while True:
        newline_index = recv_buffer.find(b"\n")
        if newline_index != -1:
            raw = bytes(recv_buffer[:newline_index])
            del recv_buffer[: newline_index + 1]
            if not raw.strip():
                continue
            return json.loads(raw.decode("utf-8"))

        chunk = conn.recv(8192)
        if not chunk:
            return None
        recv_buffer.extend(chunk)
        if len(recv_buffer) > (1 << 20):
            raise ValueError("incoming message too large")


def clamp_and_validate_action(action: np.ndarray) -> list[float]:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 8:
        raise ValueError(f"expected action dim 8, got {arr.shape[0]}")

    low = last_vector_core.Simulator.action_low()
    high = last_vector_core.Simulator.action_high()
    arr = np.clip(arr, low, high)
    arr[4] = 1.0 if arr[4] > 0.5 else 0.0
    arr[5] = 1.0 if arr[5] > 0.5 else 0.0
    arr[6] = 1.0 if arr[6] > 0.5 else 0.0
    arr[7] = float(np.clip(np.rint(arr[7]), -1, 2))
    return [float(x) for x in arr.tolist()]


def resolve_model_name(model_path: Path) -> str:
    return model_path.name


def handle_client(conn: socket.socket, model: PPO, model_name: str) -> None:
    recv_buffer = bytearray()

    hello = recv_json_line(conn, recv_buffer)
    if hello is None:
        return
    if hello.get("type") != "hello":
        raise ValueError("client did not send hello")

    conn.sendall(json_dumps_line({"type": "hello", "model": model_name}))

    while True:
        message = recv_json_line(conn, recv_buffer)
        if message is None:
            return

        obs = message.get("obs")
        if not isinstance(obs, list):
            raise ValueError("request missing obs list")

        obs_arr = np.asarray(obs, dtype=np.float32)
        action, _ = model.predict(obs_arr, deterministic=True)
        response = {"action": clamp_and_validate_action(action)}
        conn.sendall(json_dumps_line(response))


def main() -> None:
    args = parse_args()
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must be in range [1, 65535]")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    model = PPO.load(str(model_path), device="cpu")
    model_name = resolve_model_name(model_path)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)

        print(f"agent_server listening on {args.host}:{args.port} model={model_name}", flush=True)

        while True:
            conn, addr = server.accept()
            print(f"client connected: {addr[0]}:{addr[1]}", flush=True)
            with conn:
                try:
                    handle_client(conn, model=model, model_name=model_name)
                    print("client disconnected", flush=True)
                except (json.JSONDecodeError, ValueError) as exc:
                    print(f"client protocol error: {exc}", flush=True)
                except (ConnectionError, OSError) as exc:
                    print(f"client socket error: {exc}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
