#pragma once

#include "state.hpp"

#include <vector>

namespace lv {

std::vector<float> build_observation(const GameState& state);

} // namespace lv
