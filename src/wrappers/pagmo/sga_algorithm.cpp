#include "hpoea/wrappers/pagmo/sga_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <chrono>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
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
    d.integer_range = hpoea::core::IntegerRange{5, 5000};
    d.default_value = std::int64_t{50};
    d.required = true;
    space.add_descriptor(d);

    d = {};
    d.name = "generations";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 1000};
    d.default_value = std::int64_t{200};
    space.add_descriptor(d);

    d = {};
    d.name = "crossover_probability";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.9;
    space.add_descriptor(d);

    d = {};
    d.name = "mutation_probability";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.02;
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"SGA", "pagmo::sga", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoSga::PagmoSga()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoSga::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

core::OptimizationResult PagmoSga::run(const core::IProblem &problem,
                                       const core::Budget &budget,
                                       unsigned long seed) {
    const auto cr = get_double_param(configured_parameters_, "crossover_probability");
    const auto mp = get_double_param(configured_parameters_, "mutation_probability");

    return run_population(
        problem,
        budget,
        configured_parameters_,
        seed,
        [=](unsigned generations, unsigned algo_seed) {
            return pagmo::algorithm{
                pagmo::sga(generations, cr, 1.0, mp, 1.0, 2u, "exponential", "polynomial", "tournament", algo_seed)};
        });
}

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoSga::clone() const {
    return std::make_unique<PagmoSga>(*this);
}

PagmoSgaFactory::PagmoSgaFactory()
    : parameter_space_(make_parameter_space()),
      identity_(make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoSgaFactory::create() const {
    return std::make_unique<PagmoSga>();
}

} // namespace hpoea::pagmo_wrappers

