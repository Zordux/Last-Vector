#pragma once

#include <cstdint>

namespace lv {

constexpr float kFixedDt = 1.0f / 60.0f;
constexpr float kArenaWidth = 1400.0f;
constexpr float kArenaHeight = 900.0f;
constexpr int kZombieObsCount = 8;
constexpr int kRayCount = 16;
constexpr float kEpisodeLimitSeconds = 180.0f;

enum class RunMode {
    Rendered,
    Headless
};

enum class PlayState : uint8_t {
    Playing,
    ChoosingUpgrade,
    Dead
};

} // namespace lv
