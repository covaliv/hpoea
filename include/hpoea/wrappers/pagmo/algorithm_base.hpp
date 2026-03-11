#pragma once

#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"

#include <memory>

namespace hpoea::pagmo_wrappers {

// base for all pagmo EA wrappers.
// provides identity(), parameter_space(), and configure(); subclasses
// only need to implement run() and clone().
class PagmoAlgorithmBase : public core::IEvolutionaryAlgorithm {
public:
    [[nodiscard]] const core::AlgorithmIdentity &identity() const noexcept override { return identity_; }
    [[nodiscard]] const core::ParameterSpace &parameter_space() const noexcept override { return parameter_space_; }
    void configure(const core::ParameterSet &parameters) override;

protected:
    PagmoAlgorithmBase(core::ParameterSpace space, core::AlgorithmIdentity identity);

    core::ParameterSpace parameter_space_;
    core::ParameterSet configured_parameters_;
    core::AlgorithmIdentity identity_;
};

// base for all pagmo EA factories.
// provides identity() and parameter_space(); subclasses only implement create().
class PagmoAlgorithmFactoryBase : public core::IEvolutionaryAlgorithmFactory {
public:
    [[nodiscard]] const core::ParameterSpace &parameter_space() const noexcept override { return parameter_space_; }
    [[nodiscard]] const core::AlgorithmIdentity &identity() const noexcept override { return identity_; }

protected:
    PagmoAlgorithmFactoryBase(core::ParameterSpace space, core::AlgorithmIdentity identity);

    core::ParameterSpace parameter_space_;
    core::AlgorithmIdentity identity_;
};

// common parameter descriptor helpers used across multiple EA wrappers

inline core::ParameterDescriptor make_population_size_descriptor(
    std::int64_t default_val = 50,
    core::IntegerRange range = {5, 5000}) {
    core::ParameterDescriptor d;
    d.name = "population_size";
    d.type = core::ParameterType::Integer;
    d.integer_range = range;
    d.default_value = default_val;
    d.required = true;
    return d;
}

inline core::ParameterDescriptor make_generations_descriptor(
    std::int64_t default_val = 100,
    core::IntegerRange range = {1, 1000}) {
    core::ParameterDescriptor d;
    d.name = "generations";
    d.type = core::ParameterType::Integer;
    d.integer_range = range;
    d.default_value = default_val;
    return d;
}

inline core::ParameterDescriptor make_ftol_descriptor(double default_val = 1e-6) {
    core::ParameterDescriptor d;
    d.name = "ftol";
    d.type = core::ParameterType::Continuous;
    d.continuous_range = core::ContinuousRange{0.0, 1.0};
    d.default_value = default_val;
    return d;
}

inline core::ParameterDescriptor make_xtol_descriptor(double default_val = 1e-6) {
    core::ParameterDescriptor d;
    d.name = "xtol";
    d.type = core::ParameterType::Continuous;
    d.continuous_range = core::ContinuousRange{0.0, 1.0};
    d.default_value = default_val;
    return d;
}

} // namespace hpoea::pagmo_wrappers
