#pragma once

#include "hpoea/core/hyperparameter_optimizer.hpp"

#include <memory>
#include <utility>

namespace hpoea::core {

/// a no-op hyper optimizer that runs the ea once with default or fixed
/// parameters. slots into the experiment framework just like real hoas,
/// producing directly comparable runrecord/jsonl output so that
/// default-vs-tuned comparisons need no special-case code.
///
/// usage:
///   BaselineOptimizer baseline;                        // uses ea defaults
///   BaselineOptimizer baseline(custom_params);         // uses fixed params
///   manager.run_experiment(config, baseline, factory, problem, logger);
///
/// each call to optimize() runs the ea once and returns a single trial.
/// set ExperimentConfig::trials_per_optimizer to n for n independent runs.
class BaselineOptimizer final : public IHyperparameterOptimizer {
public:
    BaselineOptimizer()
        : identity_{"Baseline", "default_parameters", "1.0"} {}

    /// run the ea with these specific parameters instead of defaults.
    explicit BaselineOptimizer(ParameterSet fixed_parameters)
        : identity_{"Baseline", "fixed_parameters", "1.0"},
          fixed_parameters_(std::move(fixed_parameters)) {}

    [[nodiscard]] const AlgorithmIdentity &identity() const noexcept override {
        return identity_;
    }

    [[nodiscard]] const ParameterSpace &parameter_space() const noexcept override {
        return parameter_space_;
    }

    [[nodiscard]] HyperparameterOptimizerPtr clone() const override {
        return std::make_unique<BaselineOptimizer>(*this);
    }

    void configure(const ParameterSet & /*parameters*/) override {
        // nothing to configure, baseline has no tunable parameters
    }

    [[nodiscard]] HyperparameterOptimizationResult optimize(
        const IEvolutionaryAlgorithmFactory &algorithm_factory,
        const IProblem &problem,
        const Budget &optimizer_budget,
        const Budget &algorithm_budget,
        unsigned long seed) override;

private:
    AlgorithmIdentity identity_;
    ParameterSpace parameter_space_; // empty, no tunable parameters
    std::optional<ParameterSet> fixed_parameters_;
};

} // namespace hpoea::core
