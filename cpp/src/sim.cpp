#include "lastvector/sim.hpp"

#include "lastvector/config.hpp"
#include "lastvector/observation.hpp"
#include "lastvector/upgrade.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lv {
namespace {

constexpr float kPlayerRadius = 11.0f;
constexpr float kZombieRadius = 10.0f;

float length(Vec2 v) { return std::sqrt(v.x * v.x + v.y * v.y); }

float clampf(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }

Vec2 normalize(Vec2 v) {
    const float l = length(v);
    if (l <= 1e-6f) {
        return {0.0f, 0.0f};
    }
    return {v.x / l, v.y / l};
}

bool resolve_circle_aabb(Vec2& center, float radius, const Obstacle& o) {
    const float closest_x = clampf(center.x, o.x, o.x + o.w);
    const float closest_y = clampf(center.y, o.y, o.y + o.h);

    const float dx = center.x - closest_x;
    const float dy = center.y - closest_y;
    const float d2 = dx * dx + dy * dy;
    if (d2 >= radius * radius) {
        return false;
    }

    if (d2 > 1e-6f) {
        const float d = std::sqrt(d2);
        const float push = radius - d;
        center.x += (dx / d) * push;
        center.y += (dy / d) * push;
        return true;
    }

    const float left_pen = std::abs(center.x - o.x);
    const float right_pen = std::abs((o.x + o.w) - center.x);
    const float top_pen = std::abs(center.y - o.y);
    const float bottom_pen = std::abs((o.y + o.h) - center.y);

    float min_pen = left_pen;
    int axis = 0;
    if (right_pen < min_pen) {
        min_pen = right_pen;
        axis = 1;
    }
    if (top_pen < min_pen) {
        min_pen = top_pen;
        axis = 2;
    }
    if (bottom_pen < min_pen) {
        axis = 3;
    }

    if (axis == 0) center.x = o.x - radius;
    if (axis == 1) center.x = (o.x + o.w) + radius;
    if (axis == 2) center.y = o.y - radius;
    if (axis == 3) center.y = (o.y + o.h) + radius;
    return true;
}

void resolve_world_collisions(Vec2& center, float radius, const std::vector<Obstacle>& obstacles) {
    center.x = clampf(center.x, radius, kArenaWidth - radius);
    center.y = clampf(center.y, radius, kArenaHeight - radius);
    for (const auto& o : obstacles) {
        resolve_circle_aabb(center, radius, o);
    }
    center.x = clampf(center.x, radius, kArenaWidth - radius);
    center.y = clampf(center.y, radius, kArenaHeight - radius);
}

bool circle_hits_obstacle(const Vec2& center, float radius, const Obstacle& o) {
    const float px = clampf(center.x, o.x, o.x + o.w);
    const float py = clampf(center.y, o.y, o.y + o.h);
    const float dx = center.x - px;
    const float dy = center.y - py;
    return (dx * dx + dy * dy) <= radius * radius;
}

float nearest_zombie_distance(const GameState& state) {
    float nearest = std::numeric_limits<float>::max();
    for (const auto& z : state.zombies) {
        const float d = length({z.pos.x - state.player.pos.x, z.pos.y - state.player.pos.y});
        nearest = std::min(nearest, d);
    }
    return nearest;
}

} // namespace

Simulator::Simulator() { reset(0); }

int Simulator::observation_dim() const { return static_cast<int>(build_observation(state_).size()); }

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
        {220, 150, 180, 60},  {470, 260, 140, 50},   {640, 90, 80, 220},    {920, 170, 150, 60},
        {1080, 330, 120, 120}, {180, 420, 200, 70},  {440, 520, 60, 200},   {620, 440, 200, 80},
        {860, 560, 180, 60},   {1140, 520, 80, 200}, {250, 700, 220, 70},   {560, 760, 140, 60},
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
    if (edge == 0) z.pos = {kZombieRadius, rng_.uniform(kZombieRadius, kArenaHeight - kZombieRadius)};
    if (edge == 1) z.pos = {kArenaWidth - kZombieRadius, rng_.uniform(kZombieRadius, kArenaHeight - kZombieRadius)};
    if (edge == 2) z.pos = {rng_.uniform(kZombieRadius, kArenaWidth - kZombieRadius), kZombieRadius};
    if (edge == 3) z.pos = {rng_.uniform(kZombieRadius, kArenaWidth - kZombieRadius), kArenaHeight - kZombieRadius};
    z.hp = 26.0f + state_.difficulty_scalar * 3.0f;
    resolve_world_collisions(z.pos, kZombieRadius, state_.obstacles);
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
    if (wl > 1.0f) {
        wish = {wish.x / wl, wish.y / wl};
    }

    const float accel = 900.0f * sprint_mul;
    const float friction = 7.5f;
    p.vel.x += wish.x * accel * kFixedDt;
    p.vel.y += wish.y * accel * kFixedDt;
    p.vel.x *= (1.0f - friction * kFixedDt);
    p.vel.y *= (1.0f - friction * kFixedDt);

    const Vec2 prev_pos = p.pos;
    p.pos.x += p.vel.x * kFixedDt;
    p.pos.y += p.vel.y * kFixedDt;
    resolve_world_collisions(p.pos, kPlayerRadius, state_.obstacles);

    if (std::abs(prev_pos.x - p.pos.x) < 1e-4f) p.vel.x = 0.0f;
    if (std::abs(prev_pos.y - p.pos.y) < 1e-4f) p.vel.y = 0.0f;

    const int ext_mag = state_.upgrades.levels[static_cast<size_t>(UpgradeId::ExtendedMag)];
    p.mag_capacity = 12 + ext_mag * 3;
    if (p.mag > p.mag_capacity) {
        p.reserve += p.mag - p.mag_capacity;
        p.mag = p.mag_capacity;
    }

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
        if (length(dir) < 0.1f) {
            dir = {1.0f, 0.0f};
        }

        Bullet b{};
        b.pos = p.pos;
        b.vel = {dir.x * 760.0f, dir.y * 760.0f};

        const int big_shot = state_.upgrades.levels[static_cast<size_t>(UpgradeId::BigShot)];
        const int pierce = state_.upgrades.levels[static_cast<size_t>(UpgradeId::PiercingRounds)];
        b.radius = 4.0f + static_cast<float>(big_shot);
        b.damage = 22.0f + big_shot * 9.0f;
        b.pierce = pierce;

        state_.bullets.push_back(b);
        p.mag -= 1;
        p.shoot_cd = 0.17f + big_shot * 0.06f;
        state_.stats.shots_fired += 1;
    }
}

void Simulator::update_zombies() {
    for (auto& z : state_.zombies) {
        z.slow_timer = std::max(0.0f, z.slow_timer - kFixedDt);
        z.touch_cd = std::max(0.0f, z.touch_cd - kFixedDt);

        const Vec2 to_player{state_.player.pos.x - z.pos.x, state_.player.pos.y - z.pos.y};
        const Vec2 dir = normalize(to_player);
        float speed = 155.0f + state_.difficulty_scalar * 16.0f;
        if (z.slow_timer > 0.0f) speed *= 0.62f;

        z.vel = {dir.x * speed, dir.y * speed};
        const Vec2 prev_pos = z.pos;
        z.pos.x += z.vel.x * kFixedDt;
        z.pos.y += z.vel.y * kFixedDt;
        resolve_world_collisions(z.pos, kZombieRadius, state_.obstacles);

        if (std::abs(prev_pos.x - z.pos.x) < 1e-4f) z.vel.x = 0.0f;
        if (std::abs(prev_pos.y - z.pos.y) < 1e-4f) z.vel.y = 0.0f;
    }

    for (size_t i = 0; i < state_.zombies.size(); ++i) {
        for (size_t j = i + 1; j < state_.zombies.size(); ++j) {
            Vec2 d{state_.zombies[j].pos.x - state_.zombies[i].pos.x, state_.zombies[j].pos.y - state_.zombies[i].pos.y};
            const float l = length(d);
            const float min_sep = kZombieRadius * 2.0f;
            if (l > 1e-4f && l < min_sep) {
                const Vec2 n{d.x / l, d.y / l};
                const float push = (min_sep - l) * 0.5f;
                state_.zombies[i].pos.x -= n.x * push;
                state_.zombies[i].pos.y -= n.y * push;
                state_.zombies[j].pos.x += n.x * push;
                state_.zombies[j].pos.y += n.y * push;
                resolve_world_collisions(state_.zombies[i].pos, kZombieRadius, state_.obstacles);
                resolve_world_collisions(state_.zombies[j].pos, kZombieRadius, state_.obstacles);
            }
        }
    }
}

void Simulator::update_bullets() {
    const int frost = state_.upgrades.levels[static_cast<size_t>(UpgradeId::FrostRounds)];
    std::vector<Bullet> survivors;
    survivors.reserve(state_.bullets.size());

    for (auto b : state_.bullets) {
        b.pos.x += b.vel.x * kFixedDt;
        b.pos.y += b.vel.y * kFixedDt;

        bool destroyed = false;
        if (b.pos.x < 0.0f || b.pos.x > kArenaWidth || b.pos.y < 0.0f || b.pos.y > kArenaHeight) {
            destroyed = true;
        }

        if (!destroyed) {
            for (const auto& o : state_.obstacles) {
                if (circle_hits_obstacle(b.pos, b.radius, o)) {
                    destroyed = true;
                    break;
                }
            }
        }

        if (!destroyed) {
            for (auto& z : state_.zombies) {
                const Vec2 d{z.pos.x - b.pos.x, z.pos.y - b.pos.y};
                if (length(d) <= (kZombieRadius + b.radius)) {
                    z.hp -= b.damage;
                    if (frost > 0) {
                        z.slow_timer = std::max(z.slow_timer, 0.4f + 0.3f * frost);
                    }
                    state_.stats.shots_hit += 1;
                    b.pierce -= 1;
                    if (b.pierce < 0) {
                        destroyed = true;
                        break;
                    }
                }
            }
        }

        if (!destroyed) {
            survivors.push_back(b);
        }
    }

    state_.bullets = std::move(survivors);

    const size_t before = state_.zombies.size();
    state_.zombies.erase(
        std::remove_if(state_.zombies.begin(), state_.zombies.end(), [](const Zombie& z) { return z.hp <= 0.0f; }),
        state_.zombies.end());
    state_.stats.kills += static_cast<int>(before - state_.zombies.size());
}

void Simulator::apply_ring_of_fire() {
    const int level = state_.upgrades.levels[static_cast<size_t>(UpgradeId::RingOfFire)];
    if (level <= 0) {
        return;
    }

    const float radius = 70.0f + level * 16.0f;
    const float dps = 18.0f + level * 7.0f;
    for (auto& z : state_.zombies) {
        const Vec2 d{z.pos.x - state_.player.pos.x, z.pos.y - state_.player.pos.y};
        if (length(d) <= radius) {
            z.hp -= dps * kFixedDt;
        }
    }
}

void Simulator::handle_upgrade_choice(const Action& action) {
    if (state_.play_state != PlayState::ChoosingUpgrade) {
        return;
    }
    if (action.upgrade_choice < 0 || action.upgrade_choice > 2) {
        return;
    }

    const UpgradeId chosen = state_.upgrade_offer[static_cast<size_t>(action.upgrade_choice)];
    apply_upgrade(state_.upgrades, chosen);
    state_.play_state = PlayState::Playing;
    state_.upgrade_clock = 0.0f;
    roll_upgrade_offer();
}

float Simulator::compute_reward(const RuntimeStats& prev) const {
    (void)prev;
    return 0.02f;
}

StepResult Simulator::step(const Action& action) {
    const RuntimeStats prev_stats = state_.stats;

    handle_upgrade_choice(action);

    if (state_.play_state == PlayState::Playing) {
        update_player(action);
        update_zombies();
        update_bullets();
        apply_ring_of_fire();

        for (auto& z : state_.zombies) {
            const Vec2 d{z.pos.x - state_.player.pos.x, z.pos.y - state_.player.pos.y};
            if (length(d) < (kZombieRadius + kPlayerRadius) && z.touch_cd <= 0.0f && state_.player.invuln_timer <= 0.0f) {
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

        if (state_.player.health <= 0.0f) {
            state_.play_state = PlayState::Dead;
        }

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
    out.info.scalars["nearest_zombie_distance"] = nearest_zombie_distance(state_);
    out.info.scalars["shots_fired"] = static_cast<float>(state_.stats.shots_fired);
    out.info.scalars["shots_hit"] = static_cast<float>(state_.stats.shots_hit);
    out.info.scalars["episode_time_s"] = state_.episode_time_s;
    out.info.scalars["is_choosing_upgrade"] = state_.play_state == PlayState::ChoosingUpgrade ? 1.0f : 0.0f;
    return out;
}

} // namespace lv
