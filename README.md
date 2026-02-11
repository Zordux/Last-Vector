# Last-Vector

Deterministic 2D top-down zombie survival shooter with reinforcement-learning training (PPO + MLP) against the real C++ simulator.

## Phase 0 — Design Overview

### 1) Concise design overview
- **Simulation core (C++)**: fixed timestep deterministic world (`1/60s`) updating `GameState` only.
- **Rendering layer (raylib)**: optional frontend that reads simulation state, never mutates it.
- **Action bridge**: shared `Action` schema for human input and policy output.
- **Environment API**: `reset(seed)` / `step(action)` with deterministic RNG and upgrade-choice sub-state.
- **Python RL stack**: Gymnasium env + Stable-Baselines3 PPO (`MlpPolicy`) over flat vector observations.
- **Dashboard**: FastAPI web UI bound to `0.0.0.0` for LAN monitoring.

### 2) Folder / file tree
```text
Last-Vector/
├── LICENSE
├── README.md
├── CMakeLists.txt
├── cpp/
│   ├── include/lastvector/
│   │   ├── action.hpp
│   │   ├── config.hpp
│   │   ├── env_api.hpp
│   │   ├── observation.hpp
│   │   ├── rng.hpp
│   │   ├── sim.hpp
│   │   ├── state.hpp
│   │   └── upgrade.hpp
│   └── src/
│       ├── main.cpp
│       ├── observation.cpp
│       ├── python_bindings.cpp
│       ├── sim.cpp
│       └── upgrades.cpp
├── python/
│   ├── requirements.txt
│   ├── setup.py
│   ├── train.py
│   ├── eval.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── run_store.py
│   │   └── templates/
│   │       └── index.html
│   └── last_vector_env/
│       ├── __init__.py
│       ├── bridge.py
│       ├── env.py
│       └── reward.py
└── runs/
```

### 3) Core definitions

#### `GameState`
- Run-level: `seed`, `tick`, `episode_time_s`, `play_state`, `difficulty_scalar`.
- Player: position/velocity, health, stamina, ammo state, shoot/reload timers, invulnerability timer.
- Entities:
  - `std::vector<Zombie>` with position/velocity/hp and status timers.
  - `std::vector<Bullet>` with position/velocity/radius/damage/pierce.
  - `std::vector<Obstacle>` as static axis-aligned rectangles.
- Progression:
  - spawn budget, upgrade timer, kill/damage/shoot stats.
  - active upgrades + current 3-card offer.

#### `Action`
- `move_x ∈ [-1,1]`
- `move_y ∈ [-1,1]`
- `aim_x ∈ [-1,1]`
- `aim_y ∈ [-1,1]`
- `shoot ∈ {0,1}`
- `sprint ∈ {0,1}`
- `reload ∈ {0,1}`
- `upgrade_choice ∈ {-1,0,1,2}`

#### Observation vector (no pixels)
- Player state block.
- Nearest `N=8` zombies block.
- Ray sensors (`16` directions):
  - obstacle/boundary distance channel
  - zombie distance channel
- Difficulty and mode flags.
- Offered card IDs (3 normalized scalars).
- Active upgrade levels.

#### `StepResult`
- `observation`
- `reward`
- `terminated` (death)
- `truncated` (episode time limit)
- `info` (kills, damage, difficulty, counts, debug scalars)

### 4) Determinism, seeding, fixed timestep
- One seeded RNG (`std::mt19937_64`) per run from `reset(seed)`.
- All stochastic systems (spawns/offers) consume only this RNG.
- Simulation update uses fixed timestep (`1/60`), independent of wall-clock.
- Headless mode runs pure simulation without rendering/input side-effects.

---

## Build / Run

### C++ game
```bash
cmake -S . -B build
cmake --build build
./build/last_vector --rendered --seed 1337
```

Headless:
```bash
./build/last_vector --headless --seed 1337 --max-steps 10800
```

### Python setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
```

### Build Python extension (real C++ simulator bridge)
```bash
cd python
python setup.py build_ext --inplace
cd ..
```

### Train PPO
```bash
python python/train.py --run-id run_001 --total-steps 2000000
```

### Evaluate trained policy
```bash
python python/eval.py --model runs/run_001/best_model.zip
```

### Dashboard (LAN)
```bash
python -m python.dashboard.app --host 0.0.0.0 --port 8080 --runs-dir runs
```

Open `http://<LAN_IP>:8080`.

> Security warning: dashboard is intentionally LAN-accessible and has no authentication by default. Keep it on trusted networks only.
