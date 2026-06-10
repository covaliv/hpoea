#pragma once

#include "hpoea/core/hyper_optimizer_base.hpp"

#include <memory>

namespace hpoea::core {

class RandomSearchOptimizer final : public HyperOptimizerBase {
public:
    RandomSearchOptimizer();

    [[nodiscard]] HyperparameterOptimizerPtr clone() const override {
        return std::make_unique<RandomSearchOptimizer>(*this);
    }

    [[nodiscard]] HyperparameterOptimizationResult optimize(const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                            const IProblem &problem, const Budget &optimizer_budget,
                                                            const Budget &algorithm_budget,
                                                            unsigned long seed) override;
};

} // namespace hpoea::core
