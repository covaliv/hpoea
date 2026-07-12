#pragma once

#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
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
    std::size_t trial_index{0};
};

// trials over feval budget cannot become best
[[nodiscard]] inline bool is_selectable_trial(const HyperparameterTrialRecord &trial) {
    const auto &result = trial.optimization_result;
    if (result.status != RunStatus::Success && result.status != RunStatus::BudgetExceeded) {
        return false;
    }
    if (!std::isfinite(result.best_fitness)) {
        return false;
    }
    const auto &feval_budget = result.requested_budget.function_evaluations;
    return !feval_budget.has_value() ||
           result.algorithm_usage.function_evaluations <= *feval_budget;
}

struct HyperparameterOptimizationResult {
    RunStatus status{RunStatus::InternalError};
    ParameterSet best_parameters;
    double best_objective{std::numeric_limits<double>::infinity()};
    std::vector<HyperparameterTrialRecord> trials;
    // held out re runs of best_parameters on fresh seeds
    std::vector<OptimizationResult> validation_runs;
    OptimizerRunUsage optimizer_usage{};
    std::optional<ErrorInfo> error_info;
    unsigned long seed{0};
    ParameterSet effective_optimizer_parameters;
    std::string message;
};

class IHyperparameterOptimizer {
public:
    virtual ~IHyperparameterOptimizer() = default;

    [[nodiscard]] virtual const AlgorithmIdentity &identity() const noexcept = 0;

    [[nodiscard]] virtual const ParameterSpace &parameter_space() const noexcept = 0;

    [[nodiscard]] virtual std::unique_ptr<IHyperparameterOptimizer> clone() const = 0;

    virtual void configure(const ParameterSet &parameters) = 0;

    // values in effect
    // logged when ExperimentConfig::optimizer_parameters is unset
    [[nodiscard]] virtual const ParameterSet &configured_parameters() const noexcept {
        static const ParameterSet empty;
        return empty;
    }

    [[nodiscard]] virtual HyperparameterOptimizationResult optimize(
        const IEvolutionaryAlgorithmFactory &algorithm_factory,
        const IProblem &problem,
        const Budget &optimizer_budget,
        const Budget &algorithm_budget,
        unsigned long seed) = 0;
};

using HyperparameterOptimizerPtr = std::unique_ptr<IHyperparameterOptimizer>;

} // namespace hpoea::core
