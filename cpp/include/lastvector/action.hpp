#pragma once

namespace lv {

struct Action {
    float move_x = 0.0f;
    float move_y = 0.0f;
    float aim_x = 1.0f;
    float aim_y = 0.0f;
    bool shoot = false;
    bool sprint = false;
    bool reload = false;
    int upgrade_choice = -1;
};

} // namespace lv
