#pragma once

#include "hpoea/wrappers/pagmo/hyper_optimizer_base.hpp"

namespace hpoea::pagmo_wrappers {

class PagmoSimulatedAnnealingHyperOptimizer final : public PagmoHyperOptimizerBase {
public:
    PagmoSimulatedAnnealingHyperOptimizer();

    [[nodiscard]] core::HyperparameterOptimizerPtr clone() const override;

    [[nodiscard]] core::HyperparameterOptimizationResult optimize(
        const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
        const core::IProblem &problem,
        const core::Budget &optimizer_budget,
        const core::Budget &algorithm_budget,
        unsigned long seed) override;
};

} // namespace hpoea::pagmo_wrappers
