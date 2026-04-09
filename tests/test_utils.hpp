#pragma once

#include "hpoea/core/parameters.hpp"

#include <cmath>
#include <string>

namespace hpoea::tests_v2 {

inline bool nearly_equal(double a, double b, double tol = 1e-12) {
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    const auto diff = std::fabs(a - b);
    const auto magnitude = std::max(std::fabs(a), std::fabs(b));
    if (magnitude > 1.0) {
        return diff <= tol * magnitude;
    }
    return diff <= tol;
}

inline bool parameter_value_equals(const core::ParameterValue &lhs,
                                   const core::ParameterValue &rhs,
                                   double tol = 1e-12) {
    if (lhs.index() != rhs.index()) {
        return false;
    }
    return std::visit(
        [&](const auto &left_val) -> bool {
            using ValueType = std::decay_t<decltype(left_val)>;
            const auto &right_val = std::get<ValueType>(rhs);
            if constexpr (std::is_same_v<ValueType, double>) {
                return nearly_equal(left_val, right_val, tol);
            } else {
                return left_val == right_val;
            }
        },
        lhs);
}

inline bool parameter_set_equals(const core::ParameterSet &lhs,
                                 const core::ParameterSet &rhs,
                                 double tol = 1e-12) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (const auto &[name, value] : lhs) {
        auto it = rhs.find(name);
        if (it == rhs.end()) {
            return false;
        }
        if (!parameter_value_equals(value, it->second, tol)) {
            return false;
        }
    }
    return true;
}

}
