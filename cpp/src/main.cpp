#include "lastvector/observation.hpp"
#include "lastvector/sim.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef LASTVECTOR_WITH_RAYLIB
#include <raylib.h>
#endif

namespace {

class TcpAgentClient {
  public:
    TcpAgentClient(std::string host, std::uint16_t port) : host_(std::move(host)), port_(port) {}

    TcpAgentClient(const TcpAgentClient&) = delete;
    TcpAgentClient& operator=(const TcpAgentClient&) = delete;

    TcpAgentClient(TcpAgentClient&& other) noexcept
        : host_(std::move(other.host_)),
          port_(other.port_),
          fd_(other.fd_),
          recv_buffer_(std::move(other.recv_buffer_)) {
        other.fd_ = -1;
    }

    TcpAgentClient& operator=(TcpAgentClient&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
        host_ = std::move(other.host_);
        port_ = other.port_;
        fd_ = other.fd_;
        recv_buffer_ = std::move(other.recv_buffer_);
        other.fd_ = -1;
        return *this;
    }

    ~TcpAgentClient() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    void connect_or_throw() {
        struct addrinfo hints {};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;

        struct addrinfo* result = nullptr;
        const std::string port_str = std::to_string(port_);
        const int rv = ::getaddrinfo(host_.c_str(), port_str.c_str(), &hints, &result);
        if (rv != 0) {
            throw std::runtime_error("getaddrinfo failed: " + std::string(gai_strerror(rv)));
        }

        for (auto* rp = result; rp != nullptr; rp = rp->ai_next) {
            const int candidate = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
            if (candidate < 0) {
                continue;
            }
            if (::connect(candidate, rp->ai_addr, rp->ai_addrlen) == 0) {
                fd_ = candidate;
                break;
            }
            ::close(candidate);
        }

        ::freeaddrinfo(result);

        if (fd_ < 0) {
            throw std::runtime_error("unable to connect to agent at " + host_ + ":" + std::to_string(port_));
        }
    }

    std::string handshake_or_throw() {
        send_line_or_throw("{\"type\":\"hello\"}\n");
        const std::string line = recv_line_or_throw();
        const std::string model = extract_json_string_field(line, "model");
        if (model.empty()) {
            return "unknown";
        }
        return model;
    }

    lv::Action infer_or_throw(const std::vector<float>& obs) {
        send_line_or_throw(build_observation_json(obs));
        const std::string response = recv_line_or_throw();
        const auto action_values = parse_action_values(response);

        lv::Action action{};
        action.move_x = clamp(action_values[0], -1.0f, 1.0f);
        action.move_y = clamp(action_values[1], -1.0f, 1.0f);
        action.aim_x = clamp(action_values[2], -1.0f, 1.0f);
        action.aim_y = clamp(action_values[3], -1.0f, 1.0f);
        action.shoot = action_values[4] > 0.5f;
        action.sprint = action_values[5] > 0.5f;
        action.reload = action_values[6] > 0.5f;

        const int raw_upgrade = static_cast<int>(std::lround(action_values[7]));
        action.upgrade_choice = (raw_upgrade >= 0 && raw_upgrade <= 2) ? raw_upgrade : -1;
        return action;
    }

  private:
    static float clamp(float value, float lo, float hi) {
        return std::max(lo, std::min(hi, value));
    }

    void send_line_or_throw(const std::string& payload) {
        std::size_t sent = 0;
        while (sent < payload.size()) {
            const auto n = ::send(fd_, payload.data() + sent, payload.size() - sent, 0);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("send() failed: " + std::string(std::strerror(errno)));
            }
            if (n == 0) {
                throw std::runtime_error("send() returned 0 bytes");
            }
            sent += static_cast<std::size_t>(n);
        }
    }

    std::string recv_line_or_throw() {
        while (true) {
            const auto newline_pos = recv_buffer_.find('\n');
            if (newline_pos != std::string::npos) {
                std::string line = recv_buffer_.substr(0, newline_pos);
                recv_buffer_.erase(0, newline_pos + 1);
                return line;
            }

            char chunk[2048];
            const auto n = ::recv(fd_, chunk, sizeof(chunk), 0);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("recv() failed: " + std::string(std::strerror(errno)));
            }
            if (n == 0) {
                throw std::runtime_error("agent disconnected");
            }
            recv_buffer_.append(chunk, static_cast<std::size_t>(n));
            if (recv_buffer_.size() > 1U << 20U) {
                throw std::runtime_error("incoming message too large");
            }
        }
    }

    static std::string build_observation_json(const std::vector<float>& obs) {
        std::ostringstream oss;
        oss.precision(7);
        oss << "{\"obs\":[";
        for (std::size_t i = 0; i < obs.size(); ++i) {
            if (i > 0) {
                oss << ',';
            }
            const float value = std::isfinite(obs[i]) ? obs[i] : 0.0f;
            oss << value;
        }
        oss << "]}\n";
        return oss.str();
    }

    static std::array<float, 8> parse_action_values(const std::string& json) {
        const std::size_t key_pos = json.find("\"action\"");
        if (key_pos == std::string::npos) {
            throw std::runtime_error("agent response missing action field");
        }
        const std::size_t open = json.find('[', key_pos);
        const std::size_t close = (open == std::string::npos) ? std::string::npos : json.find(']', open);
        if (open == std::string::npos || close == std::string::npos || close <= open) {
            throw std::runtime_error("agent response has invalid action array");
        }

        std::array<float, 8> values{};
        std::size_t cursor = open + 1;
        for (std::size_t i = 0; i < values.size(); ++i) {
            while (cursor < close && std::isspace(static_cast<unsigned char>(json[cursor]))) {
                ++cursor;
            }
            if (cursor >= close) {
                throw std::runtime_error("agent action array ended early");
            }

            char* end_ptr = nullptr;
            const float parsed = std::strtof(json.c_str() + cursor, &end_ptr);
            if (end_ptr == json.c_str() + cursor || !std::isfinite(parsed)) {
                throw std::runtime_error("agent action contains non-numeric entry");
            }
            values[i] = parsed;
            cursor = static_cast<std::size_t>(end_ptr - json.c_str());

            while (cursor < close && std::isspace(static_cast<unsigned char>(json[cursor]))) {
                ++cursor;
            }

            if (i + 1 < values.size()) {
                if (cursor >= close || json[cursor] != ',') {
                    throw std::runtime_error("agent action array missing comma separator");
                }
                ++cursor;
            }
        }
        return values;
    }

    static std::string extract_json_string_field(const std::string& json, std::string_view field) {
        const std::string quoted_key = "\"" + std::string(field) + "\"";
        const std::size_t key_pos = json.find(quoted_key);
        if (key_pos == std::string::npos) {
            return {};
        }
        const std::size_t colon_pos = json.find(':', key_pos + quoted_key.size());
        if (colon_pos == std::string::npos) {
            return {};
        }
        const std::size_t q1 = json.find('"', colon_pos + 1);
        if (q1 == std::string::npos) {
            return {};
        }
        const std::size_t q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos || q2 <= q1 + 1) {
            return {};
        }
        return json.substr(q1 + 1, q2 - q1 - 1);
    }

    std::string host_;
    std::uint16_t port_ = 0;
    int fd_ = -1;
    std::string recv_buffer_;
};

struct AgentEndpoint {
    std::string host;
    std::uint16_t port = 0;
};

std::optional<AgentEndpoint> parse_agent_endpoint(const std::string& text) {
    const auto colon = text.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= text.size()) {
        return std::nullopt;
    }
    const std::string host = text.substr(0, colon);
    const std::string port_text = text.substr(colon + 1);
    try {
        const long port = std::stol(port_text);
        if (port <= 0 || port > 65535) {
            return std::nullopt;
        }
        return AgentEndpoint{host, static_cast<std::uint16_t>(port)};
    } catch (...) {
        return std::nullopt;
    }
}

void print_usage() {
    std::cout << "Usage: last_vector [--headless|--rendered] [--seed N] [--max-steps N] [--agent HOST:PORT]\n";
}

} // namespace

int main(int argc, char** argv) {
#ifdef LASTVECTOR_WITH_RAYLIB
    bool headless = false;
#endif
    std::uint64_t seed = 1337;
    int max_steps = 36000;
    std::optional<AgentEndpoint> agent_endpoint;

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
            } else if (arg == "--agent" && i + 1 < argc) {
                const auto parsed = parse_agent_endpoint(argv[++i]);
                if (!parsed.has_value()) {
                    std::cerr << "Invalid --agent endpoint. Expected HOST:PORT\n";
                    return 2;
                }
                agent_endpoint = parsed;
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

    std::string model_name = "manual";
    std::optional<TcpAgentClient> agent_client;
    if (agent_endpoint.has_value()) {
        try {
            TcpAgentClient client(agent_endpoint->host, agent_endpoint->port);
            client.connect_or_throw();
            model_name = client.handshake_or_throw();
            agent_client = std::move(client);
            std::cout << "Connected to agent server at " << agent_endpoint->host << ':' << agent_endpoint->port
                      << " model=" << model_name << '\n';
        } catch (const std::exception& ex) {
            std::cerr << "Failed to connect to agent server: " << ex.what() << '\n';
            return 2;
        }
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
            if (agent_client.has_value()) {
                try {
                    const auto obs = lv::build_observation(sim.state());
                    action = agent_client->infer_or_throw(obs);
                } catch (const std::exception& ex) {
                    std::cerr << "Agent inference failed: " << ex.what() << '\n';
                    break;
                }
            } else {
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

            if (agent_client.has_value()) {
                DrawRectangle(14, 42, 430, 138, Fade(BLACK, 0.65f));
                DrawText("AI MODE", 24, 50, 26, SKYBLUE);
                DrawText(TextFormat("Model: %s", model_name.c_str()), 24, 80, 18, LIGHTGRAY);
                DrawText(TextFormat("HP: %.1f", s.player.health), 24, 104, 18, WHITE);
                DrawText(TextFormat("Kills: %d", s.stats.kills), 24, 126, 18, WHITE);
                DrawText(TextFormat("Time Alive: %.1fs", s.episode_time_s), 24, 148, 18, WHITE);
                DrawText(TextFormat("Difficulty: %.2f", s.difficulty_scalar), 24, 170, 18, WHITE);
            }

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
        if (agent_client.has_value()) {
            try {
                const auto obs = lv::build_observation(sim.state());
                action = agent_client->infer_or_throw(obs);
            } catch (const std::exception& ex) {
                std::cerr << "Agent inference failed: " << ex.what() << '\n';
                return 2;
            }
        } else if (sim.state().play_state == lv::PlayState::ChoosingUpgrade) {
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
