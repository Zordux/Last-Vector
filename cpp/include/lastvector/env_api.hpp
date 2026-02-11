#pragma once

#include "action.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace lv {

struct StepInfo {
    int kills = 0;
    float damage_taken = 0.0f;
    int shots_fired = 0;
    int hits = 0;
    float accuracy = 0.0f;
    float damage_dealt = 0.0f;
    int selected_upgrade = -1;
    std::unordered_map<std::string, float> scalars;
};

struct StepResult {
    std::vector<float> observation;
    float reward = 0.0f;
    bool terminated = false;
    bool truncated = false;
    StepInfo info;
};

} // namespace lv
