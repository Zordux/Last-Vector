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
    out.move_x = std::clamp(a(0), -1.0f, 1.0f);
    out.move_y = std::clamp(a(1), -1.0f, 1.0f);
    out.aim_x = std::clamp(a(2), -1.0f, 1.0f);
    out.aim_y = std::clamp(a(3), -1.0f, 1.0f);
    out.shoot = a(4) >= 0.5f;
    out.sprint = a(5) >= 0.5f;
    out.reload = a(6) >= 0.5f;

    const float raw_choice = a(7);
    if (raw_choice < -0.5f) {
        out.upgrade_choice = -1;
    } else {
        out.upgrade_choice = static_cast<int>(std::round(std::clamp(raw_choice, 0.0f, 2.0f)));
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
        if (action.ndim() != 1 || action.shape(0) != lv::Simulator::action_dim()) {
            throw py::value_error("action must be a float32 array with shape (8,)");
        }
        const lv::Action parsed = action_from_array(action, sim_.state());
        auto out = sim_.step(parsed);
        steps_ += 1;

        out.truncated = out.truncated || (steps_ >= episode_steps_);

        py::dict info;
        const auto& state = sim_.state();
        info["time_alive_seconds"] = state.episode_time_s;
        info["kills"] = out.info.kills;
        info["damage_taken"] = out.info.damage_taken;
        info["shots_fired"] = out.info.shots_fired;
        info["hits"] = out.info.hits;
        info["accuracy"] = out.info.accuracy;
        info["damage_dealt"] = out.info.damage_dealt;
        info["is_choosing_upgrade"] = state.play_state == lv::PlayState::ChoosingUpgrade;

        for (const auto& [key, value] : out.info.scalars) {
            info[py::str(key)] = value;
        }

        auto obs = as_numpy(out.observation);
        auto obs_view = obs.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < obs_view.shape(0); ++i) {
            if (!std::isfinite(obs_view(i))) {
                obs_view(i) = 0.0f;
            }
        }

        float reward = out.reward;
        if (!std::isfinite(reward)) {
            reward = 0.0f;
        }

        return py::make_tuple(obs, reward, out.terminated, out.truncated, info);
    }

    int obs_dim() const { return lv::Simulator::observation_dim(); }
    int action_dim() const { return lv::Simulator::action_dim(); }

    static py::array_t<float> action_low() {
        py::array_t<float> arr(8);
        auto a = arr.mutable_unchecked<1>();
        a(0) = -1.0f; // move_x
        a(1) = -1.0f; // move_y
        a(2) = -1.0f; // aim_x
        a(3) = -1.0f; // aim_y
        a(4) = 0.0f;  // shoot
        a(5) = 0.0f;  // sprint
        a(6) = 0.0f;  // reload
        a(7) = -1.0f; // upgrade_choice
        return arr;
    }

    static py::array_t<float> action_high() {
        py::array_t<float> arr(8);
        auto a = arr.mutable_unchecked<1>();
        a(0) = 1.0f; // move_x
        a(1) = 1.0f; // move_y
        a(2) = 1.0f; // aim_x
        a(3) = 1.0f; // aim_y
        a(4) = 1.0f; // shoot
        a(5) = 1.0f; // sprint
        a(6) = 1.0f; // reload
        a(7) = 2.0f; // upgrade_choice
        return arr;
    }

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
        .def("action_dim", &PySimulator::action_dim)
        .def_static("action_low", &PySimulator::action_low)
        .def_static("action_high", &PySimulator::action_high);
}
