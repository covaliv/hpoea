#pragma once

#include "hpoea/core/hyperparameter_optimizer.hpp"

namespace hpoea::pagmo_wrappers {

class PagmoCmaesHyperOptimizer final : public core::IHyperparameterOptimizer {
public:
    PagmoCmaesHyperOptimizer();

    [[nodiscard]] const core::AlgorithmIdentity &identity() const noexcept override { return identity_; }

    [[nodiscard]] const core::ParameterSpace &parameter_space() const noexcept override { return parameter_space_; }

    void configure(const core::ParameterSet &parameters) override;

    [[nodiscard]] core::HyperparameterOptimizationResult optimize(
        const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
        const core::IProblem &problem,
        const core::Budget &budget,
        unsigned long seed) override;

private:
    core::ParameterSpace parameter_space_;
    core::ParameterSet configured_parameters_;
    core::AlgorithmIdentity identity_{};
};

} // namespace hpoea::pagmo_wrappers

