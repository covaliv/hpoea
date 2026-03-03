#include "hpoea/wrappers/pagmo/de_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <chrono>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::OptimizationResult;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor d;
    d.name = "population_size";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{5, 2000};
    d.default_value = std::int64_t{50};
    d.required = true;
    space.add_descriptor(d);

    d = {};
    d.name = "crossover_rate";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.9;
    space.add_descriptor(d);

    d = {};
    d.name = "scaling_factor";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.8;
    space.add_descriptor(d);

    d = {};
    d.name = "variant";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 10};
    d.default_value = std::int64_t{2};
    space.add_descriptor(d);

    d = {};
    d.name = "generations";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 1000};
    d.default_value = std::int64_t{100};
    space.add_descriptor(d);

    d = {};
    d.name = "ftol";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 1e-6;
    space.add_descriptor(d);

    d = {};
    d.name = "xtol";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 1e-6;
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"DifferentialEvolution", "pagmo::de", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoDifferentialEvolution::PagmoDifferentialEvolution()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

PagmoDifferentialEvolution::PagmoDifferentialEvolution(const PagmoDifferentialEvolution &other) = default;

PagmoDifferentialEvolution &PagmoDifferentialEvolution::operator=(const PagmoDifferentialEvolution &other) = default;

void PagmoDifferentialEvolution::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

OptimizationResult PagmoDifferentialEvolution::run(const core::IProblem &problem,
                                                   const core::Budget &budget,
                                                   unsigned long seed) {
    const auto crossover_rate = get_double_param(configured_parameters_, "crossover_rate");
    const auto scaling_factor = get_double_param(configured_parameters_, "scaling_factor");
    const auto variant = static_cast<unsigned>(get_int_param(configured_parameters_, "variant"));
    const auto ftol = get_double_param(configured_parameters_, "ftol");
    const auto xtol = get_double_param(configured_parameters_, "xtol");

    return run_population(
        problem,
        budget,
        configured_parameters_,
        seed,
        [=](unsigned generations, unsigned algo_seed) {
            return pagmo::algorithm{
                pagmo::de(generations, scaling_factor, crossover_rate, variant, ftol, xtol, algo_seed)};
        });
}

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoDifferentialEvolution::clone() const {
    return std::make_unique<PagmoDifferentialEvolution>(*this);
}

PagmoDifferentialEvolutionFactory::PagmoDifferentialEvolutionFactory()
    : parameter_space_(make_parameter_space()),
      identity_(make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoDifferentialEvolutionFactory::create() const {
    return std::make_unique<PagmoDifferentialEvolution>();
}

} // namespace hpoea::pagmo_wrappers
