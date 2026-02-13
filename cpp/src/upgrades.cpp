#include "lastvector/upgrade.hpp"

namespace lv {

std::vector<UpgradeDef> build_upgrade_catalog() {
    return {
        {UpgradeId::RingOfFire, "Ring of Fire", false, 5},
        {UpgradeId::BigShot, "Big Shot", false, 3},
        {UpgradeId::PiercingRounds, "Piercing Rounds", false, 3},
        {UpgradeId::FrostRounds, "Frost Rounds", false, 4},
        {UpgradeId::FastHands, "Fast Hands", false, 4},
        {UpgradeId::ExtendedMag, "Extended Mag", false, 5},
        {UpgradeId::Cardio, "Cardio", false, 5},
        {UpgradeId::SecondWind, "Second Wind", true, 1},
    };
}

void apply_upgrade(UpgradeState& state, UpgradeId id) {
    // Static initialization ensures catalog is built only once
    static const std::vector<UpgradeDef> catalog = build_upgrade_catalog();
    const auto idx = static_cast<size_t>(id);
    if (state.levels[idx] >= catalog[idx].max_stacks) {
        return;
    }
    state.levels[idx] += 1;
}

} // namespace lv
