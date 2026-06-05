#pragma once

#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/search_space.hpp"

#include <memory>

namespace hpoea::core {

class RandomSearchOptimizer final : public IHyperparameterOptimizer {
public:
    RandomSearchOptimizer();
    RandomSearchOptimizer(const RandomSearchOptimizer &other);

    [[nodiscard]] const AlgorithmIdentity &identity() const noexcept override {
        return identity_;
    }

    [[nodiscard]] const ParameterSpace &parameter_space() const noexcept override {
        return parameter_space_;
    }

    [[nodiscard]] HyperparameterOptimizerPtr clone() const override {
        return std::make_unique<RandomSearchOptimizer>(*this);
    }

    void configure(const ParameterSet &parameters) override;

    void set_search_space(std::shared_ptr<SearchSpace> search_space);

    [[nodiscard]] HyperparameterOptimizationResult optimize(const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                            const IProblem &problem, const Budget &optimizer_budget,
                                                            const Budget &algorithm_budget,
                                                            unsigned long seed) override;

private:
    ParameterSpace parameter_space_;
    ParameterSet configured_parameters_;
    AlgorithmIdentity identity_;
    std::shared_ptr<SearchSpace> search_space_;
};

} // namespace hpoea::core
