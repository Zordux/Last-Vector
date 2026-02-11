#include "lastvector/sim.hpp"
#include "lastvector/collision.hpp"
#include "lastvector/config.hpp"
#include "lastvector/observation.hpp"
#include "lastvector/upgrade.hpp"

#include <algorithm>
#include <cmath>

namespace lv {
namespace {

float length(Vec2 v) { return std::sqrt(v.x * v.x + v.y * v.y); }

constexpr float kPlayerRadius = 10.0f;
constexpr float kZombieRadius = 10.0f;
constexpr float kZombieSeparationRadius = 24.0f;

Vec2 normalize(Vec2 v) {
    const float l = length(v);
    if (l <= 1e-6f) return {0.0f, 0.0f};
    return {v.x / l, v.y / l};
}

} // namespace

Simulator::Simulator() {
    reset(0);
}

std::vector<float> Simulator::reset(uint64_t seed) {
    state_ = GameState{};
    state_.seed = seed;
    rng_.reseed(seed);
    init_obstacles();
    roll_upgrade_offer();
    return build_observation(state_);
}

void Simulator::init_obstacles() {
    state_.obstacles = {
        {220, 150, 180, 60}, {470, 260, 140, 50}, {640, 90, 80, 220}, {920, 170, 150, 60},
        {1080, 330, 120, 120}, {180, 420, 200, 70}, {440, 520, 60, 200}, {620, 440, 200, 80},
        {860, 560, 180, 60}, {1140, 520, 80, 200}, {250, 700, 220, 70}, {560, 760, 140, 60},
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
        sprint_mul = 1.55f;
        p.stamina = std::max(0.0f, p.stamina - (22.0f - cardio * 2.0f) * kFixedDt);
    } else {
        p.stamina = std::min(p.max_stamina, p.stamina + (14.0f + cardio * 2.5f) * kFixedDt);
    }

    Vec2 wish{action.move_x, action.move_y};
    const float wl = length(wish);
    if (wl > 1.0f) wish = {wish.x / wl, wish.y / wl};

    const float accel = 900.0f * sprint_mul;
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
    p.pos.x = clamp(p.pos.x, 0.0f, kArenaWidth);
    p.pos.y = clamp(p.pos.y, 0.0f, kArenaHeight);

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
    const auto& p = state_.player;
    for (auto& z : state_.zombies) {
        z.slow_timer = std::max(0.0f, z.slow_timer - kFixedDt);
        z.touch_cd = std::max(0.0f, z.touch_cd - kFixedDt);

        Vec2 dir = normalize({p.pos.x - z.pos.x, p.pos.y - z.pos.y});
        float speed = 155.0f + state_.difficulty_scalar * 16.0f;
        if (z.slow_timer > 0.0f) speed *= 0.62f;
        z.vel = {dir.x * speed, dir.y * speed};
        z.pos.x += z.vel.x * kFixedDt;
        z.pos.y += z.vel.y * kFixedDt;
    }

    for (size_t i = 0; i < state_.zombies.size(); ++i) {
        for (size_t j = i + 1; j < state_.zombies.size(); ++j) {
            Vec2 d{state_.zombies[j].pos.x - state_.zombies[i].pos.x, state_.zombies[j].pos.y - state_.zombies[i].pos.y};
            float l = length(d);
            if (l > 0.001f && l < kZombieSeparationRadius) {
                Vec2 n{d.x / l, d.y / l};
                float push = (kZombieSeparationRadius - l) * 0.5f;
                state_.zombies[i].pos.x -= n.x * push;
                state_.zombies[i].pos.y -= n.y * push;
                state_.zombies[j].pos.x += n.x * push;
                state_.zombies[j].pos.y += n.y * push;
            }
        }
    }

    for (auto& z : state_.zombies) {
        for (const auto& obstacle : state_.obstacles) {
            circle_vs_aabb_resolve(z.pos, kZombieRadius, obstacle);
        }
        z.pos.x = clamp(z.pos.x, 0.0f, kArenaWidth);
        z.pos.y = clamp(z.pos.y, 0.0f, kArenaHeight);
    }
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
                z.hp -= b.damage;
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
    if (action.upgrade_choice < 0 || action.upgrade_choice > 2) return;

    UpgradeId chosen = state_.upgrade_offer[static_cast<size_t>(action.upgrade_choice)];
    apply_upgrade(state_.upgrades, chosen);
    state_.play_state = PlayState::Playing;
    state_.upgrade_clock = 0.0f;
    roll_upgrade_offer();
}

float Simulator::compute_reward(const RuntimeStats& prev) const {
    float reward = 0.02f;
    reward += static_cast<float>(state_.stats.kills - prev.kills) * 1.25f;
    reward -= (state_.stats.damage_taken - prev.damage_taken) * 0.05f;

    const float nearest = [&]() {
        float best = 9999.0f;
        for (const auto& z : state_.zombies) {
            const float d = length({z.pos.x - state_.player.pos.x, z.pos.y - state_.player.pos.y});
            best = std::min(best, d);
        }
        return best;
    }();
    if (nearest < 120.0f) reward -= (120.0f - nearest) * 0.0008f;

    const int shots_delta = state_.stats.shots_fired - prev.shots_fired;
    const int hits_delta = state_.stats.shots_hit - prev.shots_hit;
    if (shots_delta > 0 && hits_delta == 0) reward -= 0.01f * shots_delta;
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
            if (length(d) < 16.0f && z.touch_cd <= 0.0f && state_.player.invuln_timer <= 0.0f) {
                state_.player.health -= 10.0f;
                state_.stats.damage_taken += 10.0f;
                z.touch_cd = 0.25f;
                state_.player.invuln_timer = 0.45f;
            }
        }

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
    }

    StepResult out{};
    out.observation = build_observation(state_);
    out.reward = compute_reward(prev_stats);
    out.terminated = state_.play_state == PlayState::Dead;
    out.truncated = state_.episode_time_s >= kEpisodeLimitSeconds;
    out.info.kills = state_.stats.kills;
    out.info.damage_taken = state_.stats.damage_taken;
    out.info.scalars["difficulty"] = state_.difficulty_scalar;
    out.info.scalars["zombies_alive"] = static_cast<float>(state_.zombies.size());
    return out;
}

} // namespace lv
