#pragma once

#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"

#include <limits>
#include <memory>
#include <vector>

namespace hpoea::core {

class IEvolutionaryAlgorithmFactory {
public:
    virtual ~IEvolutionaryAlgorithmFactory() = default;

    [[nodiscard]] virtual EvolutionaryAlgorithmPtr create() const = 0;

    [[nodiscard]] virtual const ParameterSpace &parameter_space() const noexcept = 0;

    [[nodiscard]] virtual const AlgorithmIdentity &identity() const noexcept = 0;
};

struct HyperparameterTrialRecord {
    ParameterSet parameters;
    OptimizationResult optimization_result;
};

struct HyperparameterOptimizationResult {
    RunStatus status{RunStatus::InternalError};
    ParameterSet best_parameters;
    double best_objective{std::numeric_limits<double>::infinity()};
    std::vector<HyperparameterTrialRecord> trials;
    BudgetUsage budget_usage{};
    unsigned long seed{0};
    ParameterSet effective_optimizer_parameters;
    std::string message;
};

class IHyperparameterOptimizer {
public:
    virtual ~IHyperparameterOptimizer() = default;

    [[nodiscard]] virtual const AlgorithmIdentity &identity() const noexcept = 0;

    [[nodiscard]] virtual const ParameterSpace &parameter_space() const noexcept = 0;

    virtual void configure(const ParameterSet &parameters) = 0;

    [[nodiscard]] virtual HyperparameterOptimizationResult optimize(
        const IEvolutionaryAlgorithmFactory &algorithm_factory,
        const IProblem &problem,
        const Budget &budget,
        unsigned long seed) = 0;
};

using HyperparameterOptimizerPtr = std::unique_ptr<IHyperparameterOptimizer>;

} // namespace hpoea::core

