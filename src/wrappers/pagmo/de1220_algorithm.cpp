#include "hpoea/wrappers/pagmo/de1220_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <vector>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

// pagmo::de1220 (pDE) is distinct from jDE which is pagmo::sade variant_adptv=1

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    space.add_descriptor(hpoea::pagmo_wrappers::make_population_size_descriptor(50, {5, 5000}));
    space.add_descriptor(hpoea::pagmo_wrappers::make_generations_descriptor(200));
    space.add_descriptor(hpoea::pagmo_wrappers::make_ftol_descriptor());
    space.add_descriptor(hpoea::pagmo_wrappers::make_xtol_descriptor());

    ParameterDescriptor d;
    d.name = "variant_adaptation";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 2};
    d.default_value = std::int64_t{1};
    space.add_descriptor(d);

    d = {};
    d.name = "memory";
    d.type = ParameterType::Boolean;
    d.default_value = false;
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"DE1220", "pagmo::de1220", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoDe1220::PagmoDe1220()
    : PagmoAlgorithmBase(make_parameter_space(), make_identity()) {}

core::OptimizationResult PagmoDe1220::run(const core::IProblem &problem,
                                           const core::Budget &budget,
                                           unsigned long seed) {
    const auto ftol = get_param<double>(configured_parameters_, "ftol");
    const auto xtol = get_param<double>(configured_parameters_, "xtol");
    const auto variant_adaptation = static_cast<unsigned>(get_param<std::int64_t>(configured_parameters_, "variant_adaptation"));
    const auto memory = get_param<bool>(configured_parameters_, "memory");
    std::vector<unsigned> allowed_variants = pagmo::de1220_statics<void>::allowed_variants;

    return run_population(
        problem,
        budget,
        configured_parameters_,
        seed,
        [=, allowed_variants = std::move(allowed_variants)](unsigned generations, unsigned algo_seed) mutable {
            return pagmo::algorithm{
                pagmo::de1220(generations, allowed_variants, variant_adaptation, ftol, xtol, memory, algo_seed)};
        });
}

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoDe1220::clone() const {
    return std::make_unique<PagmoDe1220>(*this);
}

PagmoDe1220Factory::PagmoDe1220Factory()
    : PagmoAlgorithmFactoryBase(make_parameter_space(), make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoDe1220Factory::create() const {
    return std::make_unique<PagmoDe1220>();
}

} // namespace hpoea::pagmo_wrappers
