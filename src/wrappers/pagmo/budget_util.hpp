#pragma once

#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <variant>

namespace hpoea::pagmo_wrappers {

inline std::size_t get_int_param(const core::ParameterSet &params, const char *name) {
    auto it = params.find(name);
    if (it == params.end()) {
        throw std::invalid_argument(std::string("missing parameter: ") + name);
    }
    if (!std::holds_alternative<std::int64_t>(it->second)) {
        throw std::invalid_argument(std::string("parameter '") + name + "' must be integer");
    }
    auto val = std::get<std::int64_t>(it->second);
    if (val < 0) {
        throw std::invalid_argument(std::string("parameter '") + name + "' cannot be negative");
    }
    return static_cast<std::size_t>(val);
}

inline double get_double_param(const core::ParameterSet &params, const char *name) {
    auto it = params.find(name);
    if (it == params.end()) {
        throw std::invalid_argument(std::string("missing parameter: ") + name);
    }
    if (!std::holds_alternative<double>(it->second)) {
        throw std::invalid_argument(std::string("parameter '") + name + "' must be double");
    }
    return std::get<double>(it->second);
}

inline bool get_bool_param(const core::ParameterSet &params, const char *name) {
    auto it = params.find(name);
    if (it == params.end()) {
        throw std::invalid_argument(std::string("missing parameter: ") + name);
    }
    if (!std::holds_alternative<bool>(it->second)) {
        throw std::invalid_argument(std::string("parameter '") + name + "' must be boolean");
    }
    return std::get<bool>(it->second);
}

inline std::size_t compute_generations(const core::ParameterSet &params,
                                       const core::Budget &budget,
                                       std::size_t population_size) {
    if (population_size == 0) {
        throw std::invalid_argument("population_size cannot be zero");
    }

    auto gens = get_int_param(params, "generations");
    if (gens == 0) {
        throw std::invalid_argument("generations must be positive");
    }

    if (budget.generations)
        gens = std::min(gens, *budget.generations);

    if (budget.function_evaluations) {
        auto max_gens = *budget.function_evaluations / population_size;
        gens = std::min(gens, std::max<std::size_t>(max_gens, 1));
    }

    return gens;
}

inline unsigned to_seed32(unsigned long seed) {
    return static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());
}

} // namespace hpoea::pagmo_wrappers
