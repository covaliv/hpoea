#pragma once

#include <cstdint>

namespace hpoea::core {

// splitmix64 finalizer (Vigna 2017)
inline std::uint64_t splitmix64(std::uint64_t x) {
    constexpr std::uint64_t first_multiplier = 0xbf58476d1ce4e5b9ULL;
    constexpr std::uint64_t second_multiplier = 0x94d049bb133111ebULL;

    x ^= x >> 30;
    x *= first_multiplier;
    x ^= x >> 27;
    x *= second_multiplier;
    x ^= x >> 31;
    return x;
}

inline std::uint64_t derive_stream_seed(std::uint64_t seed, std::uint64_t index) {
    constexpr std::uint64_t salt_multiplier = 0x9e3779b97f4a7c15ULL;
    return splitmix64(seed ^ ((index + 1) * salt_multiplier));
}

} // namespace hpoea::core
