#include "lastvector/config.hpp"
#include "lastvector/sim.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

float clampf(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }

py::array_t<float> as_numpy(const std::vector<float>& v) {
    py::array_t<float> arr(v.size());
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
    return arr;
}

lv::Action action_from_array(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr,
                             const lv::GameState& state) {
    if (arr.ndim() != 1 || arr.shape(0) != lv::Simulator::action_dim()) {
        throw std::runtime_error("Action must be float32 array of shape (8,)");
    }

    auto a = arr.unchecked<1>();
    lv::Action out{};
    out.move_x = clampf(a(0), -1.0f, 1.0f);
    out.move_y = clampf(a(1), -1.0f, 1.0f);
    out.aim_x = clampf(a(2), -1.0f, 1.0f);
    out.aim_y = clampf(a(3), -1.0f, 1.0f);
    out.shoot = a(4) >= 0.5f;
    out.sprint = a(5) >= 0.5f;
    out.reload = a(6) >= 0.5f;

    const float raw_choice = a(7);
    if (raw_choice < -0.5f) {
        out.upgrade_choice = -1;
    } else {
        out.upgrade_choice = static_cast<int>(std::round(clampf(raw_choice, 0.0f, 2.0f)));
    }

    if (state.play_state != lv::PlayState::ChoosingUpgrade) {
        out.upgrade_choice = -1;
    }

    return out;
}

class PySimulator {
  public:
    PySimulator(std::uint64_t seed = 0, float episode_seconds = lv::kEpisodeLimitSeconds)
        : episode_steps_(std::max(1, static_cast<int>(episode_seconds / lv::kFixedDt))) {
        reset(seed);
    }

    py::array_t<float> reset(std::uint64_t seed) {
        steps_ = 0;
        return as_numpy(sim_.reset(seed));
    }

    py::tuple step(const py::array_t<float, py::array::c_style | py::array::forcecast>& action) {
        const lv::Action parsed = action_from_array(action, sim_.state());
        auto out = sim_.step(parsed);
        steps_ += 1;

        out.truncated = out.truncated || (steps_ >= episode_steps_);

        py::dict info;
        const auto& state = sim_.state();
        info["time_alive_seconds"] = state.episode_time_s;
        info["kills"] = out.info.kills;
        info["damage_taken"] = out.info.damage_taken;
        info["is_choosing_upgrade"] = state.play_state == lv::PlayState::ChoosingUpgrade;

        const int shots_fired = state.stats.shots_fired;
        const int shots_hit = state.stats.shots_hit;
        info["accuracy"] = (shots_fired > 0) ? static_cast<float>(shots_hit) / static_cast<float>(shots_fired) : 0.0f;

        for (const auto& [key, value] : out.info.scalars) {
            info[py::str(key)] = value;
        }

        return py::make_tuple(as_numpy(out.observation), out.reward, out.terminated, out.truncated, info);
    }

    int obs_dim() const { return lv::Simulator::observation_dim(); }
    int action_dim() const { return lv::Simulator::action_dim(); }

  private:
    lv::Simulator sim_{};
    int steps_ = 0;
    int episode_steps_ = 1;
};

} // namespace

PYBIND11_MODULE(last_vector_core, m) {
    py::class_<PySimulator>(m, "Simulator")
        .def(py::init<std::uint64_t, float>(), py::arg("seed") = 0, py::arg("episode_seconds") = 180.0f)
        .def("reset", &PySimulator::reset, py::arg("seed"))
        .def("step", &PySimulator::step, py::arg("action"))
        .def("obs_dim", &PySimulator::obs_dim)
        .def("action_dim", &PySimulator::action_dim);
}
