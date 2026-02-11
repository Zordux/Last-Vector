#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace lv {

enum class UpgradeId : uint8_t {
    RingOfFire,
    BigShot,
    PiercingRounds,
    FrostRounds,
    FastHands,
    ExtendedMag,
    Cardio,
    SecondWind,
    Count
};

struct UpgradeDef {
    UpgradeId id;
    const char* name;
    bool unique;
    int max_stacks;
};

struct UpgradeState {
    std::array<int, static_cast<size_t>(UpgradeId::Count)> levels{};
    bool second_wind_used = false;
};

std::vector<UpgradeDef> build_upgrade_catalog();
void apply_upgrade(UpgradeState& state, UpgradeId id);

} // namespace lv
