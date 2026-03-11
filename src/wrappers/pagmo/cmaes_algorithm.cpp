#include "hpoea/wrappers/pagmo/cmaes_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/cmaes.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    space.add_descriptor(hpoea::pagmo_wrappers::make_population_size_descriptor(50, {5, 5000}));
    space.add_descriptor(hpoea::pagmo_wrappers::make_generations_descriptor());

    ParameterDescriptor d;
    d.name = "sigma0";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1e-6, 5.0};
    d.default_value = 0.5;
    space.add_descriptor(d);

    space.add_descriptor(hpoea::pagmo_wrappers::make_ftol_descriptor());
    space.add_descriptor(hpoea::pagmo_wrappers::make_xtol_descriptor());

    return space;
}

AlgorithmIdentity make_identity() {
    return {"CMAES", "pagmo::cmaes", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoCmaes::PagmoCmaes()
    : PagmoAlgorithmBase(make_parameter_space(), make_identity()) {}

core::OptimizationResult PagmoCmaes::run(const core::IProblem &problem,
                                         const core::Budget &budget,
                                         unsigned long seed) {
    const auto sigma0 = get_param<double>(configured_parameters_, "sigma0");
    const auto ftol = get_param<double>(configured_parameters_, "ftol");
    const auto xtol = get_param<double>(configured_parameters_, "xtol");

    return run_population(
        problem,
        budget,
        configured_parameters_,
        seed,
        [=](unsigned generations, unsigned algo_seed) {
            return pagmo::algorithm{
                pagmo::cmaes(generations, -1, -1, -1, -1, sigma0, ftol, xtol,
                             true, false, algo_seed)};
        });
}

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoCmaes::clone() const {
    return std::make_unique<PagmoCmaes>(*this);
}

PagmoCmaesFactory::PagmoCmaesFactory()
    : PagmoAlgorithmFactoryBase(make_parameter_space(), make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoCmaesFactory::create() const {
    return std::make_unique<PagmoCmaes>();
}

} // namespace hpoea::pagmo_wrappers
