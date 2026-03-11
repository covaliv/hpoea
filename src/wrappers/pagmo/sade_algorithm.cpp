#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

// pagmo::sade self-adaptation variants:
// variant_adptv=1: jDE (Brest et al., 2006)
// variant_adptv=2: iDE (Elsayed et al., 2011)

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    space.add_descriptor(hpoea::pagmo_wrappers::make_population_size_descriptor(50, {5, 2000}));
    space.add_descriptor(hpoea::pagmo_wrappers::make_generations_descriptor());

    ParameterDescriptor d;
    d.name = "variant";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 18};
    d.default_value = std::int64_t{2};
    space.add_descriptor(d);

    d = {};
    d.name = "variant_adptv";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 2};
    d.default_value = std::int64_t{1};
    space.add_descriptor(d);

    space.add_descriptor(hpoea::pagmo_wrappers::make_ftol_descriptor());
    space.add_descriptor(hpoea::pagmo_wrappers::make_xtol_descriptor());

    d = {};
    d.name = "memory";
    d.type = ParameterType::Boolean;
    d.default_value = false;
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"SelfAdaptiveDE", "pagmo::sade", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoSelfAdaptiveDE::PagmoSelfAdaptiveDE()
    : PagmoAlgorithmBase(make_parameter_space(), make_identity()) {}

core::OptimizationResult PagmoSelfAdaptiveDE::run(const core::IProblem &problem,
                                                   const core::Budget &budget,
                                                   unsigned long seed) {
    const auto variant = static_cast<unsigned>(get_param<std::int64_t>(configured_parameters_, "variant"));
    const auto variant_adptv = static_cast<unsigned>(get_param<std::int64_t>(configured_parameters_, "variant_adptv"));
    const auto ftol = get_param<double>(configured_parameters_, "ftol");
    const auto xtol = get_param<double>(configured_parameters_, "xtol");
    const auto memory = get_param<bool>(configured_parameters_, "memory");

    return run_population(
        problem,
        budget,
        configured_parameters_,
        seed,
        [=](unsigned generations, unsigned algo_seed) {
            return pagmo::algorithm{
                pagmo::sade(generations, variant, variant_adptv, ftol, xtol,
                            memory, algo_seed)};
        });
}

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoSelfAdaptiveDE::clone() const {
    return std::make_unique<PagmoSelfAdaptiveDE>(*this);
}

PagmoSelfAdaptiveDEFactory::PagmoSelfAdaptiveDEFactory()
    : PagmoAlgorithmFactoryBase(make_parameter_space(), make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoSelfAdaptiveDEFactory::create() const {
    return std::make_unique<PagmoSelfAdaptiveDE>();
}

} // namespace hpoea::pagmo_wrappers
