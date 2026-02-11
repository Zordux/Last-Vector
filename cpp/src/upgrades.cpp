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
    const auto idx = static_cast<size_t>(id);
    if (id == UpgradeId::SecondWind && state.levels[idx] > 0) {
        return;
    }
    state.levels[idx] += 1;
}

} // namespace lv
