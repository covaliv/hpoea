#include "hpoea/wrappers/pagmo/sga_algorithm.hpp"

#include "hpoea/core/problem.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>

#include "problem_adapter.hpp"

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::Budget;
using hpoea::core::OptimizationResult;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSet;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

constexpr std::string_view kFamily = "SGA";
constexpr std::string_view kImplementation = "pagmo::sga";
constexpr std::string_view kVersion = "2.x";

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor population_size;
    population_size.name = "population_size";
    population_size.type = ParameterType::Integer;
    population_size.integer_range = hpoea::core::IntegerRange{5, 5000};
    population_size.default_value = static_cast<std::int64_t>(50);
    population_size.required = true;
    space.add_descriptor(population_size);

    ParameterDescriptor generations;
    generations.name = "generations";
    generations.type = ParameterType::Integer;
    generations.integer_range = hpoea::core::IntegerRange{1, 1000};
    generations.default_value = static_cast<std::int64_t>(200);
    space.add_descriptor(generations);

    ParameterDescriptor cr;
    cr.name = "crossover_probability";
    cr.type = ParameterType::Continuous;
    cr.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    cr.default_value = 0.9;
    space.add_descriptor(cr);

    ParameterDescriptor m;
    m.name = "mutation_probability";
    m.type = ParameterType::Continuous;
    m.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    m.default_value = 0.02;
    space.add_descriptor(m);

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

PagmoSga::PagmoSga()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoSga::configure(const ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

core::OptimizationResult PagmoSga::run(const core::IProblem &problem,
                                       const core::Budget &budget,
                                       unsigned long seed) {
    OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        const auto population_size = static_cast<std::size_t>(std::get<std::int64_t>(configured_parameters_.at("population_size")));
        const auto cr = std::get<double>(configured_parameters_.at("crossover_probability"));
        const auto mp = std::get<double>(configured_parameters_.at("mutation_probability"));

        auto effective_parameters = configured_parameters_;
        const auto generations = determine_generations(configured_parameters_, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        pagmo::algorithm algorithm{pagmo::sga(static_cast<unsigned>(generations), cr, 1.0, mp, 1.0)};
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


