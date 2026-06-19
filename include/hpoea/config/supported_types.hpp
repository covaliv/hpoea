#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace hpoea::config {

// pagmo type ids shared by validator and cli dispatch
// they only gate build-dependent diagnostics
constexpr std::array<std::string_view, 6> pagmo_algorithm_type_ids{
    "de",
    "pso",
    "sade",
    "sga",
    "de1220",
    "cmaes"
};

constexpr std::array<std::string_view, 4> pagmo_optimizer_type_ids{
    "cmaes",
    "pso",
    "simulated_annealing",
    "nelder_mead"
};

template <std::size_t Size>
bool contains(const std::array<std::string_view, Size> &ids,
              std::string_view type_id) noexcept {
    for (const auto id : ids) {
        if (id == type_id) {
            return true;
        }
    }
    return false;
}

constexpr bool build_has_pagmo() noexcept {
#if defined(HPOEA_CONFIG_HAS_PAGMO)
    return true;
#else
    return false;
#endif
}

} // namespace hpoea::config
