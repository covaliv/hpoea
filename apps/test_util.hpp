#pragma once

#include "hpoea/core/parameters.hpp"

#include <cstddef>
#include <ostream>
#include <string_view>
#include <variant>
#include <vector>

inline bool in_bounds(const std::vector<double> &x,
                      const std::vector<double> &lo,
                      const std::vector<double> &hi) {
    if (x.size() != lo.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i] < lo[i] || x[i] > hi[i]) return false;
    return true;
}

namespace hpoea::apps {

inline void write_parameter_value(std::ostream &out,
                                  const core::ParameterValue &value) {
    std::visit([&out](auto v) { out << v; }, value);
}

inline void print_parameters(std::ostream &out,
                             const core::ParameterSet &parameters,
                             std::string_view indent = "") {
    for (const auto &[name, value] : parameters) {
        out << indent << name << ": ";
        write_parameter_value(out, value);
        out << "\n";
    }
}

inline void print_parameters_inline(std::ostream &out,
                                    const core::ParameterSet &parameters) {
    for (const auto &[name, value] : parameters) {
        out << name << "=";
        write_parameter_value(out, value);
        out << " ";
    }
}

} // namespace hpoea::apps
