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
    d.default_value = std::int64_t{200};
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
}

core::OptimizationResult PagmoDe1220::run(const core::IProblem &problem,
                                       const core::Budget &budget,
                                       unsigned long seed) {
    OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        const auto population_size = get_int_param(configured_parameters_, "population_size");
        const auto ftol = get_double_param(configured_parameters_, "ftol");
        const auto xtol = get_double_param(configured_parameters_, "xtol");
        const auto variant_adaptation = static_cast<unsigned>(get_int_param(configured_parameters_, "variant_adaptation"));
        const auto memory = get_bool_param(configured_parameters_, "memory");

        auto effective_parameters = configured_parameters_;
        const auto generations = compute_generations(configured_parameters_, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        std::vector<unsigned> allowed_variants = pagmo::de1220_statics<void>::allowed_variants;
        pagmo::algorithm algorithm{pagmo::de1220(static_cast<unsigned>(generations), allowed_variants, variant_adaptation, ftol, xtol, memory)};
        pagmo::problem pg_problem{ProblemAdapter{problem}};
        pagmo::population population{pg_problem, population_size, to_seed32(seed)};

        const auto start_time = std::chrono::steady_clock::now();
        population = algorithm.evolve(population);
        const auto end_time = std::chrono::steady_clock::now();

        result.best_fitness = population.champion_f()[0];
        result.best_solution = population.champion_x();
        result.budget_usage.function_evaluations = population_size * (generations + 1);
        result.budget_usage.generations = generations;
        result.budget_usage.wall_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.effective_parameters = std::move(effective_parameters);

        if (budget.wall_time.has_value() && result.budget_usage.wall_time > budget.wall_time.value()) {
            result.status = core::RunStatus::BudgetExceeded;
            result.message = "wall-time budget exceeded";
        } else {
            result.status = core::RunStatus::Success;
            result.message = "optimization completed";
        }
    } catch (const std::exception &ex) {
        result.status = core::RunStatus::InternalError;
        result.message = ex.what();
    }

    return result;
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


