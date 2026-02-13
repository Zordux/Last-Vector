#include "lastvector/sim.hpp"
#include "lastvector/collision.hpp"
#include "lastvector/config.hpp"
#include "lastvector/observation.hpp"
#include "lastvector/upgrade.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace lv {
namespace {

float length(Vec2 v) { return std::sqrt(v.x * v.x + v.y * v.y); }

constexpr float kZombieSeparationRadius = 22.0f;
constexpr float kSprintSpeedMultiplier = 1.75f;
constexpr float kMaxSeparationCorrectionPerTick = 4.0f;

Vec2 normalize(Vec2 v) {
    const float l = length(v);
    if (l <= 1e-6f) return {0.0f, 0.0f};
    return {v.x / l, v.y / l};
}

bool is_finite_vec(Vec2 v) {
    return std::isfinite(v.x) && std::isfinite(v.y);
}

Vec2 fallback_normal_for_pair(size_t a, size_t b) {
    const uint32_t bits = static_cast<uint32_t>((a * 73856093u) ^ (b * 19349663u));
    const float angle = static_cast<float>(bits % 1024u) * (6.28318530718f / 1024.0f);
    return {std::cos(angle), std::sin(angle)};
}

void clamp_position_in_bounds(Vec2& pos, float radius) {
    pos.x = std::clamp(pos.x, radius, kArenaWidth - radius);
    pos.y = std::clamp(pos.y, radius, kArenaHeight - radius);
}

void sanitize_position(Vec2& pos, Vec2 fallback, float radius) {
    if (!is_finite_vec(pos)) {
        pos = fallback;
    }
    clamp_position_in_bounds(pos, radius);
}

} // namespace

Simulator::Simulator() {
    reset(0);
}

std::vector<float> Simulator::reset(uint64_t seed) {
    state_ = GameState{};
    state_.seed = seed;
    rng_.reseed(seed);
    upgrade_pause_ticks_ = 0;
    init_obstacles();
    roll_upgrade_offer();
    return build_observation(state_);
}

void Simulator::init_obstacles() {
    const float sx = kArenaWidth / 1400.0f;
    const float sy = kArenaHeight / 900.0f;
    state_.obstacles = {
        {220.0f * sx, 150.0f * sy, 180.0f * sx, 60.0f * sy},
        {470.0f * sx, 260.0f * sy, 140.0f * sx, 50.0f * sy},
        {640.0f * sx, 90.0f * sy, 80.0f * sx, 220.0f * sy},
        {920.0f * sx, 170.0f * sy, 150.0f * sx, 60.0f * sy},
        {1080.0f * sx, 330.0f * sy, 120.0f * sx, 120.0f * sy},
        {180.0f * sx, 420.0f * sy, 200.0f * sx, 70.0f * sy},
        {440.0f * sx, 520.0f * sy, 60.0f * sx, 200.0f * sy},
        {620.0f * sx, 440.0f * sy, 200.0f * sx, 80.0f * sy},
        {860.0f * sx, 560.0f * sy, 180.0f * sx, 60.0f * sy},
        {1140.0f * sx, 520.0f * sy, 80.0f * sx, 200.0f * sy},
        {250.0f * sx, 700.0f * sy, 220.0f * sx, 70.0f * sy},
        {560.0f * sx, 760.0f * sy, 140.0f * sx, 60.0f * sy},
    };
}

void Simulator::roll_upgrade_offer() {
    for (int i = 0; i < 3; ++i) {
        state_.upgrade_offer[i] = static_cast<UpgradeId>(rng_.uniform_int(0, static_cast<int>(UpgradeId::Count) - 1));
    }
}

void Simulator::spawn_zombie() {
    Zombie z{};
    const int edge = rng_.uniform_int(0, 3);
    if (edge == 0) { z.pos = {0.0f, rng_.uniform(0.0f, kArenaHeight)}; }
    if (edge == 1) { z.pos = {kArenaWidth, rng_.uniform(0.0f, kArenaHeight)}; }
    if (edge == 2) { z.pos = {rng_.uniform(0.0f, kArenaWidth), 0.0f}; }
    if (edge == 3) { z.pos = {rng_.uniform(0.0f, kArenaWidth), kArenaHeight}; }
    z.hp = 26.0f + state_.difficulty_scalar * 3.0f;
    state_.zombies.push_back(z);
}

void Simulator::update_player(const Action& action) {
    auto& p = state_.player;
    p.shoot_cd = std::max(0.0f, p.shoot_cd - kFixedDt);
    p.reload_timer = std::max(0.0f, p.reload_timer - kFixedDt);
    p.invuln_timer = std::max(0.0f, p.invuln_timer - kFixedDt);

    float sprint_mul = 1.0f;
    const int cardio = state_.upgrades.levels[static_cast<size_t>(UpgradeId::Cardio)];
    p.max_stamina = 100.0f + cardio * 12.0f;
    if (action.sprint && p.stamina > 1.0f) {
        sprint_mul = kSprintSpeedMultiplier;
        p.stamina = std::max(0.0f, p.stamina - (22.0f - cardio * 2.0f) * kFixedDt);
    } else {
        p.stamina = std::min(p.max_stamina, p.stamina + (14.0f + cardio * 2.5f) * kFixedDt);
    }

    Vec2 wish{action.move_x, action.move_y};
    const float wl = length(wish);
    if (wl > 1.0f) wish = {wish.x / wl, wish.y / wl};

    const float accel = 930.0f * sprint_mul;
    const float friction = 7.5f;
    p.vel.x += wish.x * accel * kFixedDt;
    p.vel.y += wish.y * accel * kFixedDt;
    p.vel.x *= (1.0f - friction * kFixedDt);
    p.vel.y *= (1.0f - friction * kFixedDt);

    p.pos.x += p.vel.x * kFixedDt;
    p.pos.y += p.vel.y * kFixedDt;
    for (const auto& obstacle : state_.obstacles) {
        circle_vs_aabb_resolve(p.pos, kPlayerRadius, obstacle);
    }
    sanitize_position(p.pos, {kPlayerSpawnX, kPlayerSpawnY}, kPlayerRadius);

    const int ext_mag = state_.upgrades.levels[static_cast<size_t>(UpgradeId::ExtendedMag)];
    p.mag_capacity = 12 + ext_mag * 3;

    const int fast_hands = state_.upgrades.levels[static_cast<size_t>(UpgradeId::FastHands)];
    const float reload_time = std::max(0.35f, 1.2f - fast_hands * 0.15f);

    if (action.reload && p.reload_timer <= 0.0f && p.mag < p.mag_capacity && p.reserve > 0) {
        p.reload_timer = reload_time;
    }

    if (p.reload_timer <= 0.0f && p.mag < p.mag_capacity && p.reserve > 0) {
        const int need = p.mag_capacity - p.mag;
        const int moved = std::min(need, p.reserve);
        p.mag += moved;
        p.reserve -= moved;
    }

    if (action.shoot && p.shoot_cd <= 0.0f && p.reload_timer <= 0.0f && p.mag > 0) {
        Vec2 dir = normalize({action.aim_x, action.aim_y});
        if (length(dir) < 0.1f) dir = {1.0f, 0.0f};

        Bullet b{};
        b.pos = p.pos;
        b.vel = {dir.x * 760.0f, dir.y * 760.0f};

        const int big_shot = state_.upgrades.levels[static_cast<size_t>(UpgradeId::BigShot)];
        const int pierce = state_.upgrades.levels[static_cast<size_t>(UpgradeId::PiercingRounds)];
        b.radius = 4.0f + big_shot * 1.0f;
        b.damage = 22.0f + big_shot * 9.0f;
        b.pierce = pierce;

        state_.bullets.push_back(b);
        p.mag -= 1;
        p.shoot_cd = 0.17f + big_shot * 0.06f;
        state_.stats.shots_fired += 1;
    }
}

void Simulator::update_zombies() {
    auto& p = state_.player;
    for (auto& z : state_.zombies) {
        z.slow_timer = std::max(0.0f, z.slow_timer - kFixedDt);
        z.touch_cd = std::max(0.0f, z.touch_cd - kFixedDt);

        Vec2 dir = normalize({p.pos.x - z.pos.x, p.pos.y - z.pos.y});
        float speed = 155.0f + state_.difficulty_scalar * 16.0f;
        if (z.slow_timer > 0.0f) speed *= 0.62f;
        z.vel = {dir.x * speed, dir.y * speed};
        z.pos.x += z.vel.x * kFixedDt;
        z.pos.y += z.vel.y * kFixedDt;
        sanitize_position(z.pos, p.pos, kZombieRadius);
    }

    for (int it = 0; it < 2; ++it) {
        for (size_t i = 0; i < state_.zombies.size(); ++i) {
            for (size_t j = i + 1; j < state_.zombies.size(); ++j) {
                Vec2 d{state_.zombies[j].pos.x - state_.zombies[i].pos.x, state_.zombies[j].pos.y - state_.zombies[i].pos.y};
                float l = length(d);
                Vec2 n{};
                if (l > 1e-6f) {
                    n = {d.x / l, d.y / l};
                } else {
                    n = fallback_normal_for_pair(i, j);
                    l = 0.0f;
                }

                if (l < kZombieSeparationRadius) {
                    const float penetration = kZombieSeparationRadius - l;
                    const float push = std::min(0.5f * penetration, kMaxSeparationCorrectionPerTick);
                    state_.zombies[i].pos.x -= n.x * push;
                    state_.zombies[i].pos.y -= n.y * push;
                    state_.zombies[j].pos.x += n.x * push;
                    state_.zombies[j].pos.y += n.y * push;
                }

                sanitize_position(state_.zombies[i].pos, p.pos, kZombieRadius);
                sanitize_position(state_.zombies[j].pos, p.pos, kZombieRadius);
            }
        }

        for (size_t i = 0; i < state_.zombies.size(); ++i) {
            auto& z = state_.zombies[i];
            Vec2 d{z.pos.x - p.pos.x, z.pos.y - p.pos.y};
            float l = length(d);
            const float min_dist = kPlayerRadius + kZombieRadius;
            if (l < min_dist) {
                Vec2 n{};
                if (l > 1e-6f) {
                    n = {d.x / l, d.y / l};
                } else {
                    n = fallback_normal_for_pair(i, state_.zombies.size() + 1);
                    l = 0.0f;
                }

                const float penetration = min_dist - l;
                const float z_push = std::min(0.9f * penetration, kMaxSeparationCorrectionPerTick);
                const float p_push = std::min(0.1f * penetration, 1.2f);
                z.pos.x += n.x * z_push;
                z.pos.y += n.y * z_push;
                p.pos.x -= n.x * p_push;
                p.pos.y -= n.y * p_push;

                sanitize_position(z.pos, p.pos, kZombieRadius);
                sanitize_position(p.pos, {kPlayerSpawnX, kPlayerSpawnY}, kPlayerRadius);
            }
        }
    }

    for (auto& z : state_.zombies) {
        for (const auto& obstacle : state_.obstacles) {
            circle_vs_aabb_resolve(z.pos, kZombieRadius, obstacle);
        }
        sanitize_position(z.pos, p.pos, kZombieRadius);
    }
    sanitize_position(p.pos, {kPlayerSpawnX, kPlayerSpawnY}, kPlayerRadius);
}

void Simulator::update_bullets() {
    const int frost = state_.upgrades.levels[static_cast<size_t>(UpgradeId::FrostRounds)];

    for (auto& b : state_.bullets) {
        b.pos.x += b.vel.x * kFixedDt;
        b.pos.y += b.vel.y * kFixedDt;

        bool hit_obstacle = false;
        for (const auto& obstacle : state_.obstacles) {
            if (circle_vs_aabb_overlap(b.pos, b.radius, obstacle)) {
                b.pos = {-1000.0f, -1000.0f};
                hit_obstacle = true;
                break;
            }
        }
        if (hit_obstacle) continue;

        for (auto& z : state_.zombies) {
            Vec2 d{z.pos.x - b.pos.x, z.pos.y - b.pos.y};
            if (length(d) <= (10.0f + b.radius)) {
                const float damage_applied = std::min(z.hp, b.damage);
                z.hp -= b.damage;
                state_.stats.damage_dealt += std::max(0.0f, damage_applied);
                if (frost > 0) z.slow_timer = std::max(z.slow_timer, 0.4f + 0.3f * frost);
                b.pierce -= 1;
                state_.stats.shots_hit += 1;
                if (b.pierce < 0) {
                    b.pos = {-1000.0f, -1000.0f};
                    break;
                }
            }
        }
    }

    state_.bullets.erase(
        std::remove_if(state_.bullets.begin(), state_.bullets.end(), [](const Bullet& b) {
            return b.pos.x < 0.0f || b.pos.y < 0.0f || b.pos.x > kArenaWidth || b.pos.y > kArenaHeight;
        }),
        state_.bullets.end());

    const size_t prev = state_.zombies.size();
    state_.zombies.erase(std::remove_if(state_.zombies.begin(), state_.zombies.end(), [](const Zombie& z) { return z.hp <= 0.0f; }),
                         state_.zombies.end());
    state_.stats.kills += static_cast<int>(prev - state_.zombies.size());
}

void Simulator::apply_ring_of_fire() {
    const int level = state_.upgrades.levels[static_cast<size_t>(UpgradeId::RingOfFire)];
    if (level <= 0) return;
    const float radius = 70.0f + level * 16.0f;
    const float dps = 18.0f + level * 7.0f;
    for (auto& z : state_.zombies) {
        Vec2 d{z.pos.x - state_.player.pos.x, z.pos.y - state_.player.pos.y};
        if (length(d) < radius) z.hp -= dps * kFixedDt;
    }
}

void Simulator::handle_upgrade_choice(const Action& action) {
    if (state_.play_state != PlayState::ChoosingUpgrade) return;
    
    bool valid_choice = (action.upgrade_choice >= 0 && action.upgrade_choice <= 2);
    
    if (!valid_choice) {
        upgrade_pause_ticks_ += 1;
        if (upgrade_pause_ticks_ < kUpgradeChoiceTimeoutTicks) {
            return;
        }
    }
    
    int choice_index = valid_choice ? action.upgrade_choice : 0;
    UpgradeId chosen = state_.upgrade_offer[static_cast<size_t>(choice_index)];
    apply_upgrade(state_.upgrades, chosen);
    state_.play_state = PlayState::Playing;
    state_.upgrade_clock = 0.0f;
    upgrade_pause_ticks_ = 0;
    roll_upgrade_offer();
}

float Simulator::compute_reward(const RuntimeStats& prev) const {
    float reward = 0.02f;
    const int kills_delta = state_.stats.kills - prev.kills;
    const float damage_taken_delta = state_.stats.damage_taken - prev.damage_taken;
    const int shots_delta = state_.stats.shots_fired - prev.shots_fired;
    const int hits_delta = state_.stats.shots_hit - prev.shots_hit;
    const float damage_dealt_delta = state_.stats.damage_dealt - prev.damage_dealt;

    reward += static_cast<float>(kills_delta) * 1.45f;
    reward += static_cast<float>(hits_delta) * 0.03f;
    reward += damage_dealt_delta * 0.002f;
    reward -= damage_taken_delta * 0.05f;

    const float nearest = [&]() {
        float best = 9999.0f;
        for (const auto& z : state_.zombies) {
            const float d = length({z.pos.x - state_.player.pos.x, z.pos.y - state_.player.pos.y});
            best = std::min(best, d);
        }
        return best;
    }();
    if (nearest < 120.0f) reward -= (120.0f - nearest) * 0.0008f;

    if (shots_delta > 0 && hits_delta == 0) reward -= 0.008f * shots_delta;
    return reward;
}

StepResult Simulator::step(const Action& action) {
    RuntimeStats prev_stats = state_.stats;

    handle_upgrade_choice(action);

    if (state_.play_state == PlayState::Playing) {
        update_player(action);
        update_zombies();
        update_bullets();
        apply_ring_of_fire();

        for (auto& z : state_.zombies) {
            Vec2 d{z.pos.x - state_.player.pos.x, z.pos.y - state_.player.pos.y};
            if (length(d) < (kPlayerRadius + kZombieRadius) && z.touch_cd <= 0.0f && state_.player.invuln_timer <= 0.0f) {
                state_.player.health -= 10.0f;
                state_.stats.damage_taken += 10.0f;
                z.touch_cd = 1.5f;
            }
        }

        state_.player.health = std::max(0.0f, state_.player.health);

        if (state_.player.health <= 0.0f) {
            const size_t sw = static_cast<size_t>(UpgradeId::SecondWind);
            if (state_.upgrades.levels[sw] > 0 && !state_.upgrades.second_wind_used) {
                state_.upgrades.second_wind_used = true;
                state_.player.health = state_.player.max_health * 0.6f;
                state_.player.invuln_timer = 2.0f;
            }
        }

        if (state_.player.health <= 0.0f) state_.play_state = PlayState::Dead;

        state_.difficulty_scalar = state_.episode_time_s / 90.0f;
        const float spawn_rate = 1.0f + state_.difficulty_scalar * 1.2f;
        const int max_alive = 16 + static_cast<int>(state_.difficulty_scalar * 18.0f);
        state_.spawn_budget += spawn_rate * kFixedDt;
        while (state_.spawn_budget > 1.0f && static_cast<int>(state_.zombies.size()) < max_alive) {
            state_.spawn_budget -= 1.0f;
            spawn_zombie();
        }

        state_.upgrade_clock += kFixedDt;
        if (state_.upgrade_clock >= 20.0f) {
            state_.play_state = PlayState::ChoosingUpgrade;
        }

        state_.episode_time_s += kFixedDt;
        state_.tick += 1;

#ifndef NDEBUG
        assert(is_finite_vec(state_.player.pos));
        assert(is_finite_vec(state_.player.vel));
        assert(state_.player.health >= 0.0f);
        assert(state_.player.stamina >= 0.0f);
        for (const auto& z : state_.zombies) {
            assert(is_finite_vec(z.pos));
            assert(is_finite_vec(z.vel));
            assert(z.touch_cd >= 0.0f);
            assert(z.slow_timer >= 0.0f);
        }
#endif
    }

    StepResult out{};
    out.observation = build_observation(state_);
    out.reward = compute_reward(prev_stats);
    out.terminated = state_.play_state == PlayState::Dead;
    out.truncated = state_.episode_time_s >= kEpisodeLimitSeconds;
    out.info.kills = state_.stats.kills;
    out.info.damage_taken = state_.stats.damage_taken;
    out.info.shots_fired = state_.stats.shots_fired;
    out.info.hits = state_.stats.shots_hit;
    out.info.accuracy = (state_.stats.shots_fired > 0)
                            ? static_cast<float>(state_.stats.shots_hit) / static_cast<float>(state_.stats.shots_fired)
                            : 0.0f;
    out.info.damage_dealt = state_.stats.damage_dealt;
    out.info.scalars["difficulty"] = state_.difficulty_scalar;
    out.info.scalars["zombies_alive"] = static_cast<float>(state_.zombies.size());
    out.info.scalars["shots_fired"] = static_cast<float>(state_.stats.shots_fired);
    out.info.scalars["hits"] = static_cast<float>(state_.stats.shots_hit);
    out.info.scalars["accuracy"] = out.info.accuracy;
    out.info.scalars["damage_dealt"] = state_.stats.damage_dealt;
    out.info.scalars["kills"] = static_cast<float>(state_.stats.kills);
    out.info.scalars["damage_taken"] = state_.stats.damage_taken;
    return out;
}

} // namespace lv
