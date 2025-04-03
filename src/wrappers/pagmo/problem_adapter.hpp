#pragma once

#include "hpoea/core/problem.hpp"

#include <pagmo/types.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace hpoea::pagmo_wrappers {

class ProblemAdapter {
public:
    ProblemAdapter() = default;

    explicit ProblemAdapter(const hpoea::core::IProblem &problem) : problem_(&problem) {}

    [[nodiscard]] pagmo::vector_double fitness(const pagmo::vector_double &decision_vector) const {
        return {problem().evaluate(decision_vector)};
    }

    [[nodiscard]] std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
        const auto &reference = problem();
        auto lower = reference.lower_bounds();
        auto upper = reference.upper_bounds();
        return {pagmo::vector_double(lower.begin(), lower.end()), pagmo::vector_double(upper.begin(), upper.end())};
    }

    [[nodiscard]] bool has_gradient() const { return false; }

    [[nodiscard]] bool has_hessians() const { return false; }

    [[nodiscard]] std::string get_name() const { return problem().metadata().id; }

    [[nodiscard]] bool is_stochastic() const { return problem().is_stochastic(); }

private:
    [[nodiscard]] const hpoea::core::IProblem &problem() const {
        if (problem_ == nullptr) {
            throw std::runtime_error("ProblemAdapter used without associated problem instance");
        }
        return *problem_;
    }

    const hpoea::core::IProblem *problem_{nullptr};
};

} // namespace hpoea::pagmo_wrappers

