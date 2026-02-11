#include "lastvector/observation.hpp"
#include "lastvector/collision.hpp"
#include "lastvector/config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lv {

namespace {
float len(Vec2 v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

constexpr float kZombieRadius = 10.0f;
constexpr float kRayMaxRange = 600.0f;
constexpr float kTwoPi = 6.28318530718f;

float normalize_ray_t(float t_hit) {
    if (!std::isfinite(t_hit)) return 1.0f;
    return clamp(t_hit / kRayMaxRange, 0.0f, 1.0f);
}
} // namespace

std::vector<float> build_observation(const GameState& state) {
    std::vector<float> obs;
    obs.reserve(16 + kZombieObsCount * 5 + (kRayCount * 2) + 36);

    const auto& p = state.player;
    obs.push_back(p.pos.x / kArenaWidth);
    obs.push_back(p.pos.y / kArenaHeight);
    obs.push_back(p.vel.x / 400.0f);
    obs.push_back(p.vel.y / 400.0f);
    obs.push_back(p.health / p.max_health);
    obs.push_back(p.stamina / std::max(1.0f, p.max_stamina));
    obs.push_back(static_cast<float>(p.mag) / std::max(1, p.mag_capacity));
    obs.push_back(static_cast<float>(p.reserve) / 300.0f);
    obs.push_back(p.shoot_cd);
    obs.push_back(p.reload_timer);
    obs.push_back(p.invuln_timer);

    std::vector<std::pair<float, const Zombie*>> near;
    near.reserve(state.zombies.size());
    for (const auto& z : state.zombies) {
        const float dx = z.pos.x - p.pos.x;
        const float dy = z.pos.y - p.pos.y;
        near.push_back({dx * dx + dy * dy, &z});
    }
    std::sort(near.begin(), near.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    for (int i = 0; i < kZombieObsCount; ++i) {
        if (i < static_cast<int>(near.size())) {
            const Zombie& z = *near[i].second;
            const Vec2 rel{z.pos.x - p.pos.x, z.pos.y - p.pos.y};
            obs.push_back(rel.x / kArenaWidth);
            obs.push_back(rel.y / kArenaHeight);
            obs.push_back(len(rel) / 500.0f);
            obs.push_back((z.vel.x - p.vel.x) / 400.0f);
            obs.push_back((z.vel.y - p.vel.y) / 400.0f);
        } else {
            obs.insert(obs.end(), {0, 0, 1, 0, 0});
        }
    }

    const Obstacle arena_bounds{0.0f, 0.0f, kArenaWidth, kArenaHeight};
    for (int i = 0; i < kRayCount; ++i) {
        const float theta = (static_cast<float>(i) / static_cast<float>(kRayCount)) * kTwoPi;
        const Vec2 dir{std::cos(theta), std::sin(theta)};

        float obstacle_t = ray_intersect_aabb(p.pos, dir, arena_bounds);
        for (const auto& obstacle : state.obstacles) {
            obstacle_t = std::min(obstacle_t, ray_intersect_aabb(p.pos, dir, obstacle));
        }

        float zombie_t = std::numeric_limits<float>::infinity();
        for (const auto& z : state.zombies) {
            zombie_t = std::min(zombie_t, ray_intersect_circle(p.pos, dir, z.pos, kZombieRadius));
        }

        obs.push_back(normalize_ray_t(std::min(obstacle_t, kRayMaxRange)));
        obs.push_back(normalize_ray_t(std::min(zombie_t, kRayMaxRange)));
    }

    obs.push_back(state.difficulty_scalar);
    const bool choosing_upgrade = state.play_state == PlayState::ChoosingUpgrade;
    obs.push_back(choosing_upgrade ? 1.0f : 0.0f);

    if (choosing_upgrade) {
        const float denom = std::max(1.0f, static_cast<float>(static_cast<int>(UpgradeId::Count) - 1));
        for (int i = 0; i < 3; ++i) {
            const int upgrade_id = static_cast<int>(state.upgrade_offer[static_cast<size_t>(i)]);
            obs.push_back(static_cast<float>(upgrade_id) / denom);
        }
    } else {
        obs.insert(obs.end(), {0.0f, 0.0f, 0.0f});
    }

    for (int lv : state.upgrades.levels) {
        obs.push_back(static_cast<float>(lv) / 5.0f);
    }

    return obs;
}

} // namespace lv
