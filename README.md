# Last-Vector

Last-Vector is a deterministic zombie-survival simulator with:
- C++ simulation core
- Python Gymnasium + SB3 PPO training loop
- FastAPI dashboard for LAN monitoring

## Linux prerequisites

- `cmake` (>= 3.16)
- `g++` with C++20 support
- `python3` + `python3-venv`
- `pip`
- NVIDIA driver/CUDA stack (for GPU PyTorch on Linux laptop)

Optional:
- `raylib` development package (for rendered game executable)

---

## Build (C++ game + Python extension)

From repo root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If your environment blocks GitHub access, install `pybind11` into the active venv and point CMake to it:

```bash
python -m pip install pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
cmake --build build -j
```

This builds:
- `last_vector` executable
- `last_vector_core` Python module at:
  `python/last_vector_env/native/last_vector_core.so`

If needed, add native module folder to `PYTHONPATH`:

```bash
export PYTHONPATH="$(pwd)/python/last_vector_env/native:${PYTHONPATH}"
```

---

## Python setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

If you need explicit GPU torch wheels, install torch first per your CUDA version, then install the rest.

---

## Train PPO

```bash
export PYTHONPATH="$(pwd)/python:$(pwd)/python/last_vector_env/native:${PYTHONPATH}"
python python/train.py --run-id run_001 --total-steps 2000000
```

Artifacts under `runs/run_001/`:
- `config.json`
- `metrics.csv`
- `checkpoints/`
- `best_model.zip`
- `tensorboard/`

Launch TensorBoard:

```bash
tensorboard --logdir runs/run_001/tensorboard
```

---

## Evaluate policy

```bash
export PYTHONPATH="$(pwd)/python:$(pwd)/python/last_vector_env/native:${PYTHONPATH}"
python python/eval.py --model runs/run_001/best_model.zip --episodes 5
```

Evaluation runs headless and prints aggregate metrics.

---

## Watch trained PPO in rendered game (inference only)

Terminal 1 (agent server):

```bash
export PYTHONPATH="$(pwd)/python:$(pwd)/python/last_vector_env/native:${PYTHONPATH}"
python python/agent_server.py --model runs/test2/best_model.zip --host 127.0.0.1 --port 5555
```

Terminal 2 (rendered game client):

```bash
./build/last_vector --agent 127.0.0.1:5555 --seed 0
```

`--agent HOST:PORT` switches control to the inference server and disables local player input.

---

## Dashboard (LAN)

Run from repo root:

```bash
python -m python.dashboard.app --host 0.0.0.0 --port 8080 --runs-dir runs
```

Open: `http://<your-lan-ip>:8080`

### LAN security warning

The dashboard is intentionally bound to `0.0.0.0` and has no auth. Only expose it on trusted LANs.
