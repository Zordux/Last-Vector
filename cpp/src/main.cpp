#include "lastvector/sim.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

#ifdef LASTVECTOR_WITH_RAYLIB
#include <raylib.h>
#endif

namespace {

void print_usage() {
    std::cout << "Usage: last_vector [--headless|--rendered] [--seed N] [--max-steps N]\n";
}

} // namespace

int main(int argc, char** argv) {
#ifdef LASTVECTOR_WITH_RAYLIB
    bool headless = false;
#endif
    std::uint64_t seed = 1337;
    int max_steps = 36000;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
#ifndef LASTVECTOR_WITH_RAYLIB
            if (arg == "--rendered") {
                std::cerr << "Rendered mode is unavailable: built without raylib.\n";
                return 2;
            }
#endif
            if (arg == "--headless") {
#ifdef LASTVECTOR_WITH_RAYLIB
                headless = true;
#endif
            } else if (arg == "--rendered") {
#ifdef LASTVECTOR_WITH_RAYLIB
                headless = false;
#endif
            } else if (arg == "--seed" && i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else if (arg == "--max-steps" && i + 1 < argc) {
                max_steps = std::stoi(argv[++i]);
            } else if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            } else {
                std::cerr << "Unknown or incomplete argument: " << arg << '\n';
                print_usage();
                return 2;
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "Failed to parse CLI arguments: " << ex.what() << '\n';
        return 2;
    }

    if (max_steps < 1) {
        std::cerr << "--max-steps must be >= 1\n";
        return 2;
    }

    lv::Simulator sim;
    sim.reset(seed);

#ifdef LASTVECTOR_WITH_RAYLIB
    if (!headless) {
        InitWindow(1280, 720, "Last-Vector");
        SetTargetFPS(60);

        Camera2D camera{};
        camera.offset = {GetScreenWidth() * 0.5f, GetScreenHeight() * 0.5f};
        camera.target = {lv::kPlayerSpawnX, lv::kPlayerSpawnY};
        camera.rotation = 0.0f;
        camera.zoom = 1.0f;

        while (!WindowShouldClose()) {
            lv::Action action{};
            action.move_x = (IsKeyDown(KEY_D) ? 1.0f : 0.0f) - (IsKeyDown(KEY_A) ? 1.0f : 0.0f);
            action.move_y = (IsKeyDown(KEY_S) ? 1.0f : 0.0f) - (IsKeyDown(KEY_W) ? 1.0f : 0.0f);
            action.sprint = IsKeyDown(KEY_LEFT_SHIFT);
            action.reload = IsKeyPressed(KEY_R);
            action.shoot = IsMouseButtonDown(MOUSE_BUTTON_LEFT);

            const auto& state = sim.state();
            const Vector2 mouse_world = GetScreenToWorld2D(GetMousePosition(), camera);
            action.aim_x = (mouse_world.x - state.player.pos.x) / 300.0f;
            action.aim_y = (mouse_world.y - state.player.pos.y) / 300.0f;

            if (state.play_state == lv::PlayState::ChoosingUpgrade) {
                if (IsKeyPressed(KEY_ONE)) action.upgrade_choice = 0;
                if (IsKeyPressed(KEY_TWO)) action.upgrade_choice = 1;
                if (IsKeyPressed(KEY_THREE)) action.upgrade_choice = 2;
            }

            sim.step(action);

            const auto& s = sim.state();
            const lv::Vec2 look_dir{action.aim_x, action.aim_y};
            const float look_len = std::sqrt(look_dir.x * look_dir.x + look_dir.y * look_dir.y);
            lv::Vec2 look_n = look_dir;
            if (look_len > 1e-5f && std::isfinite(look_len)) {
                look_n.x /= look_len;
                look_n.y /= look_len;
            } else {
                look_n = {0.0f, 0.0f};
            }

            const Vector2 desired_target{
                s.player.pos.x + look_n.x * lv::kCameraLookAheadDistance,
                s.player.pos.y + look_n.y * lv::kCameraLookAheadDistance,
            };
            camera.target.x += (desired_target.x - camera.target.x) * lv::kCameraFollowLerp;
            camera.target.y += (desired_target.y - camera.target.y) * lv::kCameraFollowLerp;

            BeginDrawing();
            ClearBackground(BLACK);

            BeginMode2D(camera);
            DrawRectangleLinesEx({0.0f, 0.0f, lv::kArenaWidth, lv::kArenaHeight}, 2.0f, DARKGRAY);
            DrawCircleV({s.player.pos.x, s.player.pos.y}, 10.0f, GREEN);
            for (const auto& z : s.zombies) DrawCircleV({z.pos.x, z.pos.y}, 10.0f, RED);
            for (const auto& b : s.bullets) DrawCircleV({b.pos.x, b.pos.y}, b.radius, YELLOW);
            for (const auto& o : s.obstacles) DrawRectangleLinesEx({o.x, o.y, o.w, o.h}, 1.0f, GRAY);

            for (int i = 0; i < lv::kRayCount; ++i) {
                const float theta = (static_cast<float>(i) / static_cast<float>(lv::kRayCount)) * 6.28318530718f;
                const Vector2 ray_end{
                    s.player.pos.x + std::cos(theta) * 160.0f,
                    s.player.pos.y + std::sin(theta) * 160.0f,
                };
                DrawLineV({s.player.pos.x, s.player.pos.y}, ray_end, Fade(SKYBLUE, 0.28f));
            }
            EndMode2D();

            DrawText(TextFormat("HP %.1f  STA %.1f  MAG %d/%d  Kills %d", s.player.health, s.player.stamina, s.player.mag,
                                s.player.reserve, s.stats.kills),
                     16, 16, 20, WHITE);

            if (s.play_state == lv::PlayState::ChoosingUpgrade) {
                DrawRectangle(180, 140, 920, 440, Fade(DARKGRAY, 0.9f));
                DrawText("Choose upgrade (1/2/3)", 220, 180, 30, WHITE);
                DrawText(TextFormat("1) %d", static_cast<int>(s.upgrade_offer[0])), 220, 250, 24, GOLD);
                DrawText(TextFormat("2) %d", static_cast<int>(s.upgrade_offer[1])), 220, 300, 24, GOLD);
                DrawText(TextFormat("3) %d", static_cast<int>(s.upgrade_offer[2])), 220, 350, 24, GOLD);
            }

            EndDrawing();

            if (s.play_state == lv::PlayState::Dead) {
                break;
            }
        }

        CloseWindow();
        return 0;
    }
#endif

    for (int i = 0; i < max_steps; ++i) {
        lv::Action action{};
        if (sim.state().play_state == lv::PlayState::ChoosingUpgrade) {
            action.upgrade_choice = 0;
        }
        const auto res = sim.step(action);
        if (res.terminated || res.truncated) {
            break;
        }
    }

    const auto& end_state = sim.state();
    std::cout << "seed=" << seed << " ticks=" << end_state.tick << " kills=" << end_state.stats.kills
              << " dead=" << (end_state.play_state == lv::PlayState::Dead ? 1 : 0) << '\n';
    return 0;
}
