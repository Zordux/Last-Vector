#pragma once

#include <cstdint>
#include <random>

namespace lv {

class DeterministicRng {
  public:
    explicit DeterministicRng(uint64_t seed = 0) : eng_(seed) {}

    void reseed(uint64_t seed) { eng_.seed(seed); }

    float uniform(float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(eng_);
    }

    int uniform_int(int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(eng_);
    }

  private:
    std::mt19937_64 eng_;
};

} // namespace lv
