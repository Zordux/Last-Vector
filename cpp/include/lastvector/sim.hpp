#pragma once

#include "action.hpp"
#include "env_api.hpp"
#include "rng.hpp"
#include "state.hpp"

#include <cstdint>

namespace lv {

class Simulator {
  public:
    Simulator();

    std::vector<float> reset(uint64_t seed);
    StepResult step(const Action& action);

    const GameState& state() const { return state_; }

  private:
    GameState state_{};
    DeterministicRng rng_{0};

    void init_obstacles();
    void roll_upgrade_offer();
    void spawn_zombie();
    void update_player(const Action& action);
    void update_zombies();
    void update_bullets();
    void apply_ring_of_fire();
    void handle_upgrade_choice(const Action& action);
    float compute_reward(const RuntimeStats& prev) const;
};

} // namespace lv
