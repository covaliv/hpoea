#pragma once

#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/search_space.hpp"

#include <memory>

namespace hpoea::pagmo_wrappers {

// shared base for all pagmo-based hyper optimizers.
// handles identity, parameter space, configure (with validation), and search space.
class PagmoHyperOptimizerBase : public core::IHyperparameterOptimizer {
public:
    [[nodiscard]] const core::AlgorithmIdentity &identity() const noexcept override {
        return identity_;
    }

    [[nodiscard]] const core::ParameterSpace &parameter_space() const noexcept override {
        return parameter_space_;
    }

    void configure(const core::ParameterSet &parameters) override;

    void set_search_space(std::shared_ptr<core::SearchSpace> search_space);

protected:
    PagmoHyperOptimizerBase(core::ParameterSpace space, core::AlgorithmIdentity identity);

    core::ParameterSpace parameter_space_;
    core::ParameterSet configured_parameters_;
    core::AlgorithmIdentity identity_;
    std::shared_ptr<core::SearchSpace> search_space_;
};

} // namespace hpoea::pagmo_wrappers
