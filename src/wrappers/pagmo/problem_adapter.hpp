#pragma once

#include "hpoea/core/error_classification.hpp"
#include "hpoea/core/problem.hpp"

#include <cmath>
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
        try {
            const auto value = problem().evaluate(decision_vector);
            if (!std::isfinite(value)) {
                throw core::EvaluationFailure("problem evaluation returned non-finite value");
            }
            return {value};
        } catch (const core::EvaluationFailure &) {
            throw;
        } catch (const std::exception &ex) {
            throw core::EvaluationFailure(ex.what());
        } catch (...) {
            throw core::EvaluationFailure("problem evaluation failed with unknown error");
        }
    }

    [[nodiscard]] std::pair<pagmo::vector_double, pagmo::vector_double> get_bounds() const {
        const auto &reference = problem();
        auto lower = reference.lower_bounds();
        auto upper = reference.upper_bounds();
        if (lower.size() != upper.size()) {
            throw std::invalid_argument("lower/upper bounds dimension mismatch: " +
                std::to_string(lower.size()) + " vs " + std::to_string(upper.size()));
        }
        if (lower.size() != static_cast<std::size_t>(reference.dimension())) {
            throw std::invalid_argument("bounds dimension (" + std::to_string(lower.size()) +
                ") != problem dimension (" + std::to_string(reference.dimension()) + ")");
        }
        for (std::size_t i = 0; i < lower.size(); ++i) {
            if (lower[i] > upper[i]) {
                throw std::invalid_argument("lower bound > upper bound at dimension " + std::to_string(i));
            }
        }
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
