#pragma once

#include "hpoea/core/hyperparameter_optimizer.hpp"

#include <memory>
#include <utility>

namespace hpoea::core {

// no-op hyper optimizer
// runs the ea once per optimize() call with default or fixed parameters
// keeps default-vs-tuned comparisons on the same runrecord pipeline
class BaselineOptimizer final : public IHyperparameterOptimizer {
public:
    BaselineOptimizer()
        : identity_{"Baseline", "default_parameters", "1.0"} {}

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
        // nothing to configure
        // baseline has no tunable parameters
    }

    [[nodiscard]] const ParameterSet &configured_parameters() const noexcept override {
        return fixed_parameters_ ? *fixed_parameters_ : IHyperparameterOptimizer::configured_parameters();
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
