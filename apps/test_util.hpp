#pragma once

#include <cstddef>
#include <vector>

inline bool in_bounds(const std::vector<double> &x,
                      const std::vector<double> &lo,
                      const std::vector<double> &hi) {
    if (x.size() != lo.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i] < lo[i] || x[i] > hi[i]) return false;
    return true;
}
