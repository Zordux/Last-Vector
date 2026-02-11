#include "lastvector/sim.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring>

namespace py = pybind11;

namespace {

float clampf(float v, float lo, float hi) { return std::max(lo, std::min(v, hi)); }

lv::Action action_from_array(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
    if (arr.ndim() != 1 || arr.shape(0) != lv::Simulator::action_dim()) {
        throw std::runtime_error("Action must be float32 array of shape (8,)");
    }

    auto a = arr.unchecked<1>();
    lv::Action out{};
    out.move_x = clampf(a(0), -1.0f, 1.0f);
    out.move_y = clampf(a(1), -1.0f, 1.0f);
    out.aim_x = clampf(a(2), -1.0f, 1.0f);
    out.aim_y = clampf(a(3), -1.0f, 1.0f);
    out.shoot = a(4) > 0.5f;
    out.sprint = a(5) > 0.5f;
    out.reload = a(6) > 0.5f;

    const float raw_choice = a(7);
    if (raw_choice < -0.5f) {
        out.upgrade_choice = -1;
    } else {
        out.upgrade_choice = static_cast<int>(std::round(clampf(raw_choice, 0.0f, 2.0f)));
    }
    return out;
}

py::array_t<float> as_numpy(const std::vector<float>& v) {
    py::array_t<float> arr(v.size());
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(float));
    return arr;
}

} // namespace

PYBIND11_MODULE(last_vector_core, m) {
    py::class_<lv::Simulator>(m, "Simulator")
        .def(py::init<>())
        .def("reset", [](lv::Simulator& self, std::uint64_t seed) { return as_numpy(self.reset(seed)); })
        .def("step", [](lv::Simulator& self, const py::array_t<float, py::array::c_style | py::array::forcecast>& action) {
            const lv::Action parsed = action_from_array(action);
            const auto out = self.step(parsed);

            py::dict info;
            info["kills"] = out.info.kills;
            info["damage_taken"] = out.info.damage_taken;
            info["selected_upgrade"] = out.info.selected_upgrade;
            for (const auto& [key, value] : out.info.scalars) {
                info[py::str(key)] = value;
            }

            return py::make_tuple(as_numpy(out.observation), out.reward, out.terminated, out.truncated, info);
        })
        .def("observation_dim", &lv::Simulator::observation_dim)
        .def_static("action_dim", &lv::Simulator::action_dim);
}
