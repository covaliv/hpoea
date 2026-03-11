#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    space.add_descriptor(hpoea::pagmo_wrappers::make_population_size_descriptor(50, {5, 2000}));

    ParameterDescriptor d;
    d.name = "omega";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.7298;
    space.add_descriptor(d);

    d = {};
    d.name = "eta1";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1.0, 3.0};
    d.default_value = 2.05;
    space.add_descriptor(d);

    d = {};
    d.name = "eta2";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1.0, 3.0};
    d.default_value = 2.05;
    space.add_descriptor(d);

    d = {};
    d.name = "max_velocity";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 100.0};
    d.default_value = 0.5;
    space.add_descriptor(d);

    d = {};
    d.name = "variant";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 6};
    d.default_value = std::int64_t{5};
    space.add_descriptor(d);

    space.add_descriptor(hpoea::pagmo_wrappers::make_generations_descriptor());

    return space;
}

AlgorithmIdentity make_identity() {
    return {"ParticleSwarmOptimization", "pagmo::pso", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoParticleSwarmOptimization::PagmoParticleSwarmOptimization()
    : PagmoAlgorithmBase(make_parameter_space(), make_identity()) {}

core::OptimizationResult PagmoParticleSwarmOptimization::run(const core::IProblem &problem,
                                                             const core::Budget &budget,
                                                             unsigned long seed) {
    const auto omega = get_param<double>(configured_parameters_, "omega");
    const auto eta1 = get_param<double>(configured_parameters_, "eta1");
    const auto eta2 = get_param<double>(configured_parameters_, "eta2");
    const auto max_velocity = get_param<double>(configured_parameters_, "max_velocity");
    const auto variant = static_cast<unsigned>(get_param<std::int64_t>(configured_parameters_, "variant"));

    return run_population(
        problem,
        budget,
        configured_parameters_,
        seed,
        [=](unsigned generations, unsigned algo_seed) {
            return pagmo::algorithm{
                pagmo::pso(generations, omega, eta1, eta2, max_velocity,
                           variant, 2u, 4u, false, algo_seed)};
        });
}

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoParticleSwarmOptimization::clone() const {
    return std::make_unique<PagmoParticleSwarmOptimization>(*this);
}

PagmoParticleSwarmOptimizationFactory::PagmoParticleSwarmOptimizationFactory()
    : PagmoAlgorithmFactoryBase(make_parameter_space(), make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoParticleSwarmOptimizationFactory::create() const {
    return std::make_unique<PagmoParticleSwarmOptimization>();
}

} // namespace hpoea::pagmo_wrappers
