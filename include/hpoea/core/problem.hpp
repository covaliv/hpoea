#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace hpoea::core {

struct ProblemMetadata {
    std::string id;
    std::string family;
    std::string description;
};

class IProblem {
public:
    virtual ~IProblem() = default;

    [[nodiscard]] virtual const ProblemMetadata &metadata() const noexcept = 0;

    [[nodiscard]] virtual std::size_t dimension() const = 0;

    [[nodiscard]] virtual std::vector<double> lower_bounds() const = 0;

    [[nodiscard]] virtual std::vector<double> upper_bounds() const = 0;

    [[nodiscard]] virtual double evaluate(const std::vector<double> &decision_vector) const = 0;

    [[nodiscard]] virtual bool is_stochastic() const noexcept { return false; }
};

} // namespace hpoea::core

