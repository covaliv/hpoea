#include "hpoea/wrappers/pagmo/sga_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    space.add_descriptor(hpoea::pagmo_wrappers::make_population_size_descriptor(50, {5, 5000}));
    space.add_descriptor(hpoea::pagmo_wrappers::make_generations_descriptor(200));

    ParameterDescriptor d;
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
    : PagmoAlgorithmBase(make_parameter_space(), make_identity()) {}

core::OptimizationResult PagmoSga::run(const core::IProblem &problem,
                                       const core::Budget &budget,
                                       unsigned long seed) {
    const auto cr = get_param<double>(configured_parameters_, "crossover_probability");
    const auto mp = get_param<double>(configured_parameters_, "mutation_probability");

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
    : PagmoAlgorithmFactoryBase(make_parameter_space(), make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoSgaFactory::create() const {
    return std::make_unique<PagmoSga>();
}

} // namespace hpoea::pagmo_wrappers
