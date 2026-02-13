#pragma once

#include <cstdint>

namespace lv {

constexpr float kFixedDt = 1.0f / 60.0f;
constexpr float kArenaWidth = 4200.0f;
constexpr float kArenaHeight = 2800.0f;
constexpr float kPlayerSpawnX = kArenaWidth * 0.5f;
constexpr float kPlayerSpawnY = kArenaHeight * 0.5f;
constexpr float kCameraFollowLerp = 0.14f;
constexpr float kCameraLookAheadDistance = 96.0f;
constexpr int kZombieObsCount = 8;
constexpr int kRayCount = 16;
constexpr float kEpisodeLimitSeconds = 180.0f;
constexpr float kPlayerRadius = 10.0f;
constexpr float kZombieRadius = 10.0f;

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
