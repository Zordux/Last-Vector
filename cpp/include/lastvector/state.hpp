#pragma once

#include "config.hpp"
#include "upgrade.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace lv {

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Player {
    Vec2 pos{kPlayerSpawnX, kPlayerSpawnY};
    Vec2 vel{};
    float health = 100.0f;
    float max_health = 100.0f;
    float stamina = 100.0f;
    float max_stamina = 100.0f;
    int mag = 12;
    int mag_capacity = 12;
    int reserve = 120;
    float shoot_cd = 0.0f;
    float reload_timer = 0.0f;
    float invuln_timer = 0.0f;
};

struct Zombie {
    Vec2 pos{};
    Vec2 vel{};
    float hp = 30.0f;
    float slow_timer = 0.0f;
    float touch_cd = 0.0f;
};

struct Bullet {
    Vec2 pos{};
    Vec2 vel{};
    float radius = 4.0f;
    float damage = 22.0f;
    int pierce = 0;
};

struct Obstacle {
    float x = 0.0f;
    float y = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
};

struct RuntimeStats {
    int kills = 0;
    float damage_taken = 0.0f;
    int shots_fired = 0;
    int shots_hit = 0;
};

struct GameState {
    uint64_t seed = 0;
    uint64_t tick = 0;
    float episode_time_s = 0.0f;
    PlayState play_state = PlayState::Playing;
    float difficulty_scalar = 0.0f;

    Player player{};
    std::vector<Zombie> zombies;
    std::vector<Bullet> bullets;
    std::vector<Obstacle> obstacles;

    UpgradeState upgrades{};
    std::array<UpgradeId, 3> upgrade_offer{
        UpgradeId::RingOfFire,
        UpgradeId::BigShot,
        UpgradeId::PiercingRounds,
    };

    float spawn_budget = 0.0f;
    float upgrade_clock = 0.0f;

    RuntimeStats stats{};
};

} // namespace lv
