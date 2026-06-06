#pragma once

#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/search_space.hpp"

#include <memory>

namespace hpoea::core {

// shared base for configurable hyper optimizers
class HyperOptimizerBase : public IHyperparameterOptimizer {
public:
    [[nodiscard]] const AlgorithmIdentity &identity() const noexcept override {
        return identity_;
    }

    [[nodiscard]] const ParameterSpace &parameter_space() const noexcept override {
        return parameter_space_;
    }

    [[nodiscard]] const ParameterSet &configured_parameters() const noexcept override {
        return configured_parameters_;
    }

    void configure(const ParameterSet &parameters) override;

    void set_search_space(std::shared_ptr<SearchSpace> search_space);

protected:
    HyperOptimizerBase(ParameterSpace space, AlgorithmIdentity identity);
    HyperOptimizerBase(const HyperOptimizerBase &other);

    ParameterSpace parameter_space_;
    ParameterSet configured_parameters_;
    AlgorithmIdentity identity_;
    std::shared_ptr<SearchSpace> search_space_;
};

} // namespace hpoea::core
