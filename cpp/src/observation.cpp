#include "lastvector/observation.hpp"

#include "lastvector/config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lv {
namespace {

constexpr float kRayRange = 320.0f;
constexpr float kZombieRadius = 10.0f;

float len(Vec2 v) { return std::sqrt(v.x * v.x + v.y * v.y); }

float ray_aabb_t(Vec2 origin, Vec2 dir, const Obstacle& o) {
    const float eps = 1e-6f;

    float tmin = 0.0f;
    float tmax = kRayRange;

    if (std::abs(dir.x) < eps) {
        if (origin.x < o.x || origin.x > o.x + o.w) return -1.0f;
    } else {
        const float tx1 = (o.x - origin.x) / dir.x;
        const float tx2 = (o.x + o.w - origin.x) / dir.x;
        tmin = std::max(tmin, std::min(tx1, tx2));
        tmax = std::min(tmax, std::max(tx1, tx2));
    }

    if (std::abs(dir.y) < eps) {
        if (origin.y < o.y || origin.y > o.y + o.h) return -1.0f;
    } else {
        const float ty1 = (o.y - origin.y) / dir.y;
        const float ty2 = (o.y + o.h - origin.y) / dir.y;
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));
    }

    if (tmax < 0.0f || tmin > tmax) return -1.0f;
    return tmin;
}

float ray_circle_t(Vec2 origin, Vec2 dir, Vec2 center, float radius) {
    const Vec2 oc{origin.x - center.x, origin.y - center.y};
    const float b = 2.0f * (oc.x * dir.x + oc.y * dir.y);
    const float c = oc.x * oc.x + oc.y * oc.y - radius * radius;
    const float disc = b * b - 4.0f * c;
    if (disc < 0.0f) return -1.0f;
    const float s = std::sqrt(disc);
    const float t0 = (-b - s) * 0.5f;
    const float t1 = (-b + s) * 0.5f;
    if (t0 >= 0.0f) return t0;
    if (t1 >= 0.0f) return t1;
    return -1.0f;
}

float boundary_hit_t(Vec2 origin, Vec2 dir) {
    float best = kRayRange;
    if (dir.x > 1e-6f) best = std::min(best, (kArenaWidth - origin.x) / dir.x);
    if (dir.x < -1e-6f) best = std::min(best, (0.0f - origin.x) / dir.x);
    if (dir.y > 1e-6f) best = std::min(best, (kArenaHeight - origin.y) / dir.y);
    if (dir.y < -1e-6f) best = std::min(best, (0.0f - origin.y) / dir.y);
    return std::clamp(best, 0.0f, kRayRange);
}

} // namespace

std::vector<float> build_observation(const GameState& state) {
    std::vector<float> obs;
    obs.reserve(16 + kZombieObsCount * 5 + kRayCount * 2 + 32);

    const auto& p = state.player;
    obs.push_back(p.pos.x / kArenaWidth);
    obs.push_back(p.pos.y / kArenaHeight);
    obs.push_back(p.vel.x / 400.0f);
    obs.push_back(p.vel.y / 400.0f);
    obs.push_back(p.health / std::max(1.0f, p.max_health));
    obs.push_back(p.stamina / std::max(1.0f, p.max_stamina));
    obs.push_back(static_cast<float>(p.mag) / std::max(1, p.mag_capacity));
    obs.push_back(static_cast<float>(p.reserve) / 300.0f);
    obs.push_back(p.shoot_cd);
    obs.push_back(p.reload_timer);
    obs.push_back(p.invuln_timer);

    std::vector<std::pair<float, const Zombie*>> nearest;
    nearest.reserve(state.zombies.size());
    for (const auto& z : state.zombies) {
        const float dx = z.pos.x - p.pos.x;
        const float dy = z.pos.y - p.pos.y;
        nearest.push_back({dx * dx + dy * dy, &z});
    }
    std::sort(nearest.begin(), nearest.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    for (int i = 0; i < kZombieObsCount; ++i) {
        if (i < static_cast<int>(nearest.size())) {
            const Zombie& z = *nearest[i].second;
            const Vec2 rel{z.pos.x - p.pos.x, z.pos.y - p.pos.y};
            obs.push_back(rel.x / kArenaWidth);
            obs.push_back(rel.y / kArenaHeight);
            obs.push_back(len(rel) / 500.0f);
            obs.push_back((z.vel.x - p.vel.x) / 400.0f);
            obs.push_back((z.vel.y - p.vel.y) / 400.0f);
        } else {
            obs.insert(obs.end(), {0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
        }
    }

    for (int i = 0; i < kRayCount; ++i) {
        const float theta = (static_cast<float>(i) / static_cast<float>(kRayCount)) * 6.28318530718f;
        const Vec2 dir{std::cos(theta), std::sin(theta)};

        float obstacle_t = boundary_hit_t(p.pos, dir);
        for (const auto& o : state.obstacles) {
            const float t = ray_aabb_t(p.pos, dir, o);
            if (t >= 0.0f) obstacle_t = std::min(obstacle_t, t);
        }

        float zombie_t = kRayRange;
        for (const auto& z : state.zombies) {
            const float t = ray_circle_t(p.pos, dir, z.pos, kZombieRadius);
            if (t >= 0.0f) zombie_t = std::min(zombie_t, t);
        }

        obs.push_back(std::clamp(obstacle_t / kRayRange, 0.0f, 1.0f));
        obs.push_back(std::clamp(zombie_t / kRayRange, 0.0f, 1.0f));
    }

    obs.push_back(state.difficulty_scalar);
    const bool choosing = state.play_state == PlayState::ChoosingUpgrade;
    obs.push_back(choosing ? 1.0f : 0.0f);

    const float inv_card_count = 1.0f / static_cast<float>(UpgradeId::Count);
    obs.push_back((static_cast<float>(state.upgrade_offer[0]) + 0.5f) * inv_card_count);
    obs.push_back((static_cast<float>(state.upgrade_offer[1]) + 0.5f) * inv_card_count);
    obs.push_back((static_cast<float>(state.upgrade_offer[2]) + 0.5f) * inv_card_count);

    for (const int lv : state.upgrades.levels) {
        obs.push_back(static_cast<float>(lv) / 5.0f);
    }

    return obs;
}

} // namespace lv
