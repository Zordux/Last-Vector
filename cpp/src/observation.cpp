#include "lastvector/observation.hpp"
#include "lastvector/config.hpp"

#include <algorithm>
#include <cmath>

namespace lv {

namespace {
float len(Vec2 v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}
} // namespace

std::vector<float> build_observation(const GameState& state) {
    std::vector<float> obs;
    obs.reserve(16 + kZombieObsCount * 5 + kRayCount + 32);

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

    for (int i = 0; i < kRayCount; ++i) {
        const float theta = (static_cast<float>(i) / static_cast<float>(kRayCount)) * 6.28318530718f;
        const float dx = std::cos(theta);
        const float dy = std::sin(theta);
        float t = 1.0f;
        if (dx > 0.0f) t = std::min(t, (kArenaWidth - p.pos.x) / (dx * 300.0f));
        if (dx < 0.0f) t = std::min(t, (0.0f - p.pos.x) / (dx * 300.0f));
        if (dy > 0.0f) t = std::min(t, (kArenaHeight - p.pos.y) / (dy * 300.0f));
        if (dy < 0.0f) t = std::min(t, (0.0f - p.pos.y) / (dy * 300.0f));
        obs.push_back(std::clamp(t, 0.0f, 1.0f));
    }

    obs.push_back(state.difficulty_scalar);
    obs.push_back(state.play_state == PlayState::ChoosingUpgrade ? 1.0f : 0.0f);

    for (int lv : state.upgrades.levels) {
        obs.push_back(static_cast<float>(lv) / 5.0f);
    }

    return obs;
}

} // namespace lv
