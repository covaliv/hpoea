#include "hpoea/wrappers/pagmo/de_algorithm.hpp"

#include "hpoea/core/problem.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>

#include "problem_adapter.hpp"

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::Budget;
using hpoea::core::BudgetUsage;
using hpoea::core::OptimizationResult;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSet;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

constexpr std::string_view kFamily = "DifferentialEvolution";
constexpr std::string_view kImplementation = "pagmo::de";
constexpr std::string_view kVersion = "2.x";

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor population_size;
    population_size.name = "population_size";
    population_size.type = ParameterType::Integer;
    population_size.integer_range = hpoea::core::IntegerRange{5, 2000};
    population_size.default_value = static_cast<std::int64_t>(50);
    population_size.required = true;
    space.add_descriptor(population_size);

    ParameterDescriptor crossover_rate;
    crossover_rate.name = "crossover_rate";
    crossover_rate.type = ParameterType::Continuous;
    crossover_rate.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    crossover_rate.default_value = 0.9;
    space.add_descriptor(crossover_rate);

    ParameterDescriptor scaling_factor;
    scaling_factor.name = "scaling_factor";
    scaling_factor.type = ParameterType::Continuous;
    scaling_factor.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    scaling_factor.default_value = 0.8;
    space.add_descriptor(scaling_factor);

    ParameterDescriptor variant;
    variant.name = "variant";
    variant.type = ParameterType::Integer;
    variant.integer_range = hpoea::core::IntegerRange{1, 10};
    variant.default_value = static_cast<std::int64_t>(2);
    space.add_descriptor(variant);

    ParameterDescriptor generations;
    generations.name = "generations";
    generations.type = ParameterType::Integer;
    generations.integer_range = hpoea::core::IntegerRange{1, 100000};
    generations.default_value = static_cast<std::int64_t>(100);
    space.add_descriptor(generations);

    ParameterDescriptor ftol;
    ftol.name = "ftol";
    ftol.type = ParameterType::Continuous;
    ftol.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    ftol.default_value = 1e-6;
    space.add_descriptor(ftol);

    ParameterDescriptor xtol;
    xtol.name = "xtol";
    xtol.type = ParameterType::Continuous;
    xtol.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    xtol.default_value = 1e-6;
    space.add_descriptor(xtol);

    return space;
}

AlgorithmIdentity make_identity() {
    AlgorithmIdentity id;
    id.family = std::string{kFamily};
    id.implementation = std::string{kImplementation};
    id.version = std::string{kVersion};
    return id;
}

std::size_t determine_generations(const ParameterSet &parameters, const Budget &budget, std::size_t population) {
    std::size_t generations = static_cast<std::size_t>(std::get<std::int64_t>(parameters.at("generations")));

    if (budget.generations.has_value()) {
        generations = std::min(generations, budget.generations.value());
    }

    if (budget.function_evaluations.has_value()) {
        const std::size_t max_generations = budget.function_evaluations.value() / std::max<std::size_t>(population, 1);
        if (max_generations > 0) {
            generations = std::min(generations, max_generations);
        }
    }

    return std::max<std::size_t>(generations, 1);
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoDifferentialEvolution::PagmoDifferentialEvolution()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

PagmoDifferentialEvolution::PagmoDifferentialEvolution(const PagmoDifferentialEvolution &other) = default;

PagmoDifferentialEvolution &PagmoDifferentialEvolution::operator=(const PagmoDifferentialEvolution &other) = default;

void PagmoDifferentialEvolution::configure(const ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

OptimizationResult PagmoDifferentialEvolution::run(const core::IProblem &problem,
                                                   const core::Budget &budget,
                                                   unsigned long seed) {
    OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        const auto population_size = static_cast<std::size_t>(std::get<std::int64_t>(configured_parameters_.at("population_size")));
        const auto crossover_rate = std::get<double>(configured_parameters_.at("crossover_rate"));
        const auto scaling_factor = std::get<double>(configured_parameters_.at("scaling_factor"));
        const auto variant = static_cast<unsigned>(std::get<std::int64_t>(configured_parameters_.at("variant")));
        const auto ftol = std::get<double>(configured_parameters_.at("ftol"));
        const auto xtol = std::get<double>(configured_parameters_.at("xtol"));

        auto effective_parameters = configured_parameters_;

        const auto generations = determine_generations(configured_parameters_, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        const auto algorithm_seed = static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());
        pagmo::algorithm algorithm{pagmo::de(static_cast<unsigned>(generations), scaling_factor, crossover_rate, variant, ftol, xtol, algorithm_seed)};
        pagmo::problem pg_problem{ProblemAdapter{problem}};

        const auto population_seed = static_cast<unsigned int>(seed & std::numeric_limits<unsigned int>::max());
        pagmo::population population{pg_problem, population_size, population_seed};

        const auto start_time = std::chrono::steady_clock::now();
        population = algorithm.evolve(population);
        const auto end_time = std::chrono::steady_clock::now();

        result.status = core::RunStatus::Success;
        result.best_fitness = population.champion_f()[0];
        result.best_solution = population.champion_x();
        result.budget_usage.function_evaluations = population_size * generations;
        result.budget_usage.generations = generations;
        result.budget_usage.wall_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.effective_parameters = std::move(effective_parameters);
        result.message = "Optimization completed.";

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

