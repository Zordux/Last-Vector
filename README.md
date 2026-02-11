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
- `metrics.csv` (live scalar history: reward, episode length, kills, combat stats)
- `status.json` (live dashboard snapshot updated every ~7s or 10k steps)
- `checkpoints/`
- `best_model.zip`
- `tensorboard/`

Launch TensorBoard:

```bash
tensorboard --logdir runs/run_001/tensorboard
```

---

### Training metrics exposed

`status.json` now includes:
- `steps_done`, `total_steps`, `progress_pct`
- `fps`, `episodes_done`
- `best_model_path`, `best_score`
- `latest_reward`, `latest_ep_len`, `latest_kills`
- `device`, `last_update_time`

`metrics.csv` includes rolling combat fields used by the dashboard:
- `kills`, `shots_fired`, `hits`, `accuracy`
- `damage_dealt`, `damage_taken`

The environment `info` dictionary exposes the same combat counters (`shots_fired`, `hits`, `accuracy`, `damage_dealt`, `kills`, `damage_taken`) for logging and debugging.

### Reward shaping (combat-focused, deterministic)

To encourage fighting without destabilizing PPO:
- Kill reward increased from `1.25` to `1.45` (**+16%**).
- Added micro-rewards for successful combat:
  - `+0.03` per hit
  - `+0.002` per damage dealt
- Reduced wasted-shot penalty from `0.01` to `0.008` per shot when no hits land.

This keeps random firing discouraged, while making accurate combat slightly more attractive.

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
