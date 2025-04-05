#pragma once

#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"

#include <memory>

namespace hpoea::pagmo_wrappers {

class PagmoDifferentialEvolution final : public core::IEvolutionaryAlgorithm {
public:
    PagmoDifferentialEvolution();

    PagmoDifferentialEvolution(const PagmoDifferentialEvolution &other);

    PagmoDifferentialEvolution &operator=(const PagmoDifferentialEvolution &other);

    PagmoDifferentialEvolution(PagmoDifferentialEvolution &&) noexcept = default;
    PagmoDifferentialEvolution &operator=(PagmoDifferentialEvolution &&) noexcept = default;

    ~PagmoDifferentialEvolution() override = default;

    [[nodiscard]] const core::AlgorithmIdentity &identity() const noexcept override { return identity_; }

    [[nodiscard]] const core::ParameterSpace &parameter_space() const noexcept override { return parameter_space_; }

    void configure(const core::ParameterSet &parameters) override;

    [[nodiscard]] core::OptimizationResult run(const core::IProblem &problem,
                                               const core::Budget &budget,
                                               unsigned long seed) override;

    [[nodiscard]] std::unique_ptr<core::IEvolutionaryAlgorithm> clone() const override;

private:
    core::ParameterSpace parameter_space_;
    core::ParameterSet configured_parameters_;
    core::AlgorithmIdentity identity_{};
};

class PagmoDifferentialEvolutionFactory final : public core::IEvolutionaryAlgorithmFactory {
public:
    PagmoDifferentialEvolutionFactory();

    [[nodiscard]] core::EvolutionaryAlgorithmPtr create() const override;

    [[nodiscard]] const core::ParameterSpace &parameter_space() const noexcept override { return parameter_space_; }

    [[nodiscard]] const core::AlgorithmIdentity &identity() const noexcept override { return identity_; }

private:
    core::ParameterSpace parameter_space_;
    core::AlgorithmIdentity identity_{};
};

} // namespace hpoea::pagmo_wrappers

