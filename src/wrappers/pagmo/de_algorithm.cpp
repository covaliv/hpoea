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
    OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        const auto population_size = get_int_param(configured_parameters_, "population_size");
        const auto crossover_rate = get_double_param(configured_parameters_, "crossover_rate");
        const auto scaling_factor = get_double_param(configured_parameters_, "scaling_factor");
        const auto variant = static_cast<unsigned>(get_int_param(configured_parameters_, "variant"));
        const auto ftol = get_double_param(configured_parameters_, "ftol");
        const auto xtol = get_double_param(configured_parameters_, "xtol");

        auto effective_parameters = configured_parameters_;
        const auto generations = compute_generations(configured_parameters_, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        const auto algo_seed = to_seed32(seed);
        const auto pop_seed = derive_seed(seed, 1);
        pagmo::algorithm algorithm{pagmo::de(static_cast<unsigned>(generations), scaling_factor, crossover_rate, variant, ftol, xtol, algo_seed)};
        pagmo::problem pg_problem{ProblemAdapter{problem}};
        pagmo::population population{pg_problem, population_size, pop_seed};

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

