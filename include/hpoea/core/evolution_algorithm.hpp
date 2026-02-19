#pragma once

#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace hpoea::core {

struct OptimizationResult {
    RunStatus status{RunStatus::InternalError};
    double best_fitness{std::numeric_limits<double>::infinity()};
    std::vector<double> best_solution;
    BudgetUsage budget_usage{};
    ParameterSet effective_parameters{};
    unsigned long seed{0};
    std::string message;
};

class IEvolutionaryAlgorithm {
public:
    virtual ~IEvolutionaryAlgorithm() = default;

    [[nodiscard]] virtual const AlgorithmIdentity &identity() const noexcept = 0;

    [[nodiscard]] virtual const ParameterSpace &parameter_space() const noexcept = 0;

    virtual void configure(const ParameterSet &parameters) = 0;

    [[nodiscard]] virtual OptimizationResult run(const IProblem &problem, const Budget &budget, unsigned long seed) = 0;

    [[nodiscard]] virtual std::unique_ptr<IEvolutionaryAlgorithm> clone() const = 0;
};

using EvolutionaryAlgorithmPtr = std::unique_ptr<IEvolutionaryAlgorithm>;

} // namespace hpoea::core

