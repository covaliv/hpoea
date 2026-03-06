#include "hpoea/wrappers/pagmo/de1220_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <chrono>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de1220.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <vector>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::OptimizationResult;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

// pagmo::de1220 (pDE) is distinct from jDE which is pagmo::sade variant_adptv=1

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

    d = {};
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
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoDe1220::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

core::OptimizationResult PagmoDe1220::run(const core::IProblem &problem,
                                       const core::Budget &budget,
                                       unsigned long seed) {
    const auto ftol = get_double_param(configured_parameters_, "ftol");
    const auto xtol = get_double_param(configured_parameters_, "xtol");
    const auto variant_adaptation = static_cast<unsigned>(get_int_param(configured_parameters_, "variant_adaptation"));
    const auto memory = get_bool_param(configured_parameters_, "memory");
    static const std::vector<unsigned> allowed_variants = pagmo::de1220_statics<void>::allowed_variants;

    return run_population(
        problem,
        budget,
        configured_parameters_,
        seed,
        [=](unsigned generations, unsigned algo_seed) {
            return pagmo::algorithm{
                pagmo::de1220(generations, allowed_variants, variant_adaptation, ftol, xtol, memory, algo_seed)};
        });
}

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoDe1220::clone() const {
    return std::make_unique<PagmoDe1220>(*this);
}

PagmoDe1220Factory::PagmoDe1220Factory()
    : parameter_space_(make_parameter_space()),
      identity_(make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoDe1220Factory::create() const {
    return std::make_unique<PagmoDe1220>();
}

} // namespace hpoea::pagmo_wrappers

