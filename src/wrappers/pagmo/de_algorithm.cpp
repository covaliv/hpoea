#include "hpoea/wrappers/pagmo/de_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    space.add_descriptor(hpoea::pagmo_wrappers::make_population_size_descriptor(50, {5, 2000}));

    ParameterDescriptor d;
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

    space.add_descriptor(hpoea::pagmo_wrappers::make_generations_descriptor());
    space.add_descriptor(hpoea::pagmo_wrappers::make_ftol_descriptor());
    space.add_descriptor(hpoea::pagmo_wrappers::make_xtol_descriptor());

    return space;
}

AlgorithmIdentity make_identity() {
    return {"DifferentialEvolution", "pagmo::de", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoDifferentialEvolution::PagmoDifferentialEvolution()
    : PagmoAlgorithmBase(make_parameter_space(), make_identity()) {}

core::OptimizationResult PagmoDifferentialEvolution::run(const core::IProblem &problem,
                                                         const core::Budget &budget,
                                                         unsigned long seed) {
    const auto crossover_rate = get_param<double>(configured_parameters_, "crossover_rate");
    const auto scaling_factor = get_param<double>(configured_parameters_, "scaling_factor");
    const auto variant = static_cast<unsigned>(get_param<std::int64_t>(configured_parameters_, "variant"));
    const auto ftol = get_param<double>(configured_parameters_, "ftol");
    const auto xtol = get_param<double>(configured_parameters_, "xtol");

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
    : PagmoAlgorithmFactoryBase(make_parameter_space(), make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoDifferentialEvolutionFactory::create() const {
    return std::make_unique<PagmoDifferentialEvolution>();
}

} // namespace hpoea::pagmo_wrappers
