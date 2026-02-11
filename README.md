# Last-Vector

Deterministic 2D top-down zombie survival shooter with a reinforcement-learning pipeline (PPO + MLP) designed for headless simulation and local training.

## Phase 0 — Design Overview

### 1) Concise design overview
- **Simulation core (C++)**: fixed timestep deterministic world (`1/60s`) updating `GameState` only.
- **Rendering layer (raylib)**: optional visual frontend that reads immutable simulation snapshots.
- **Action bridge**: same `Action` schema for human keyboard/mouse and AI policy output.
- **Environment API**: reset/step semantics with deterministic seeding and upgrade-choice sub-state.
- **Training stack (Python)**: Gymnasium wrapper around simulation contract, Stable-Baselines3 PPO trainer, eval runner.
- **Dashboard (FastAPI)**: LAN-accessible monitor for runs, checkpoints, state, and latest metrics.

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
│       ├── sim.cpp
│       └── upgrades.cpp
├── python/
│   ├── requirements.txt
│   ├── train.py
│   ├── eval.py
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── run_store.py
│   │   └── templates/
│   │       └── index.html
│   └── last_vector_env/
│       ├── __init__.py
│       ├── env.py
│       ├── bridge.py
│       └── reward.py
└── runs/
```

### 3) Core definitions

#### `GameState`
- Run-level: `tick`, `episode_time_s`, `difficulty_scalar`, `run_seed`, `status`.
- Player: position/velocity, health/max_health, stamina/max_stamina, ammo_mag/ammo_reserve, shoot/reload timers, invuln timer, second wind state.
- Entities:
  - `std::vector<Zombie>` with position/velocity/hp/slow timers/contact cooldowns.
  - `std::vector<Bullet>` with position/velocity/radius/damage/pierce.
  - `std::vector<Obstacle>` static rectangles.
- Progression:
  - spawn accumulators, kill count, total damage taken, survival time.
  - upgrade timers and active upgrade levels.
  - current 3-card offer when state is `CHOOSING_UPGRADE`.

#### `Action`
Continuous + binary controls used by both human and AI:
- `move_x ∈ [-1,1]`
- `move_y ∈ [-1,1]`
- `aim_x ∈ [-1,1]`
- `aim_y ∈ [-1,1]`
- `shoot ∈ {0,1}`
- `sprint ∈ {0,1}`
- `reload ∈ {0,1}`
- `upgrade_choice ∈ {-1,0,1,2}` (`-1` when no choice requested)

#### Observation vector (no pixels)
Flattened deterministic vector:
1. Player state (normed): pos, vel, health, stamina, ammo, cooldowns, invuln.
2. Nearest `N=8` zombies: relative dx/dy, distance, relative vx/vy (zero-padded).
3. 16 ray distances around player to nearest obstacle/zombie.
4. Difficulty scalar and normalized timers.
5. Compact upgrade encoding (per-card level/flag).
6. Environment mode flags (playing / choosing upgrade).

#### `StepResult`
- `std::vector<float> observation`
- `float reward`
- `bool terminated` (player death)
- `bool truncated` (time-limit)
- `StepInfo info` (kills, damage, chosen card, debug counters)

### 4) Determinism, seeding, fixed timestep
- A single `DeterministicRng` (`std::mt19937_64`) is created from `reset(seed)`.
- All random events (spawn edge, spawn position, card offers) exclusively consume this RNG.
- Simulation advances only in fixed quanta (`dt = 1/60`) with no wall-clock dependence.
- Rendering interpolates nothing in training mode and never mutates simulation state.
- Headless mode bypasses raylib drawing/input and loops pure `step()` calls.

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

### Train PPO
```bash
python python/train.py --run-id run_001 --total-steps 2000000
```

### Watch trained policy
```bash
python python/eval.py --model runs/run_001/best_model.zip
```

### Dashboard (LAN)
```bash
python python/dashboard/app.py --host 0.0.0.0 --port 8080
```

Open `http://<LAN_IP>:8080`.

> Security warning: dashboard is intentionally LAN-accessible and has no authentication by default. Do not expose it to untrusted networks.
