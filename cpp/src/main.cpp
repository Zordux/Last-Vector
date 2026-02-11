#include "lastvector/sim.hpp"

#include <cstdint>
#include <iostream>
#include <string>

#ifdef LASTVECTOR_WITH_RAYLIB
#include <raylib.h>
#endif

int main(int argc, char** argv) {
    bool headless = false;
    uint64_t seed = 1337;
    int max_steps = 36000;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--headless") headless = true;
        if (arg == "--rendered") headless = false;
        if (arg == "--seed" && i + 1 < argc) seed = std::stoull(argv[++i]);
        if (arg == "--max-steps" && i + 1 < argc) max_steps = std::stoi(argv[++i]);
    }

    lv::Simulator sim;
    sim.reset(seed);

#ifdef LASTVECTOR_WITH_RAYLIB
    if (!headless) {
        InitWindow(1280, 720, "Last-Vector");
        SetTargetFPS(60);

        while (!WindowShouldClose()) {
            lv::Action action{};
            action.move_x = (IsKeyDown(KEY_D) ? 1.0f : 0.0f) - (IsKeyDown(KEY_A) ? 1.0f : 0.0f);
            action.move_y = (IsKeyDown(KEY_S) ? 1.0f : 0.0f) - (IsKeyDown(KEY_W) ? 1.0f : 0.0f);
            action.sprint = IsKeyDown(KEY_LEFT_SHIFT);
            action.reload = IsKeyPressed(KEY_R);
            action.shoot = IsMouseButtonDown(MOUSE_BUTTON_LEFT);

            Vector2 mouse = GetMousePosition();
            const auto& state = sim.state();
            action.aim_x = (mouse.x - state.player.pos.x) / 300.0f;
            action.aim_y = (mouse.y - state.player.pos.y) / 300.0f;

            if (state.play_state == lv::PlayState::ChoosingUpgrade) {
                if (IsKeyPressed(KEY_ONE)) action.upgrade_choice = 0;
                if (IsKeyPressed(KEY_TWO)) action.upgrade_choice = 1;
                if (IsKeyPressed(KEY_THREE)) action.upgrade_choice = 2;
            }

            sim.step(action);

            BeginDrawing();
            ClearBackground(BLACK);

            const auto& s = sim.state();
            DrawCircleV({s.player.pos.x, s.player.pos.y}, 10.0f, GREEN);
            for (const auto& z : s.zombies) DrawCircleV({z.pos.x, z.pos.y}, 10.0f, RED);
            for (const auto& b : s.bullets) DrawCircleV({b.pos.x, b.pos.y}, b.radius, YELLOW);
            for (const auto& o : s.obstacles) DrawRectangleLinesEx({o.x, o.y, o.w, o.h}, 1.0f, GRAY);

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
        if (res.terminated || res.truncated) break;
    }

    const auto& end_state = sim.state();
    std::cout << "seed=" << seed << " ticks=" << end_state.tick << " kills=" << end_state.stats.kills
              << " dead=" << (end_state.play_state == lv::PlayState::Dead ? 1 : 0) << '\n';
    return 0;
}
