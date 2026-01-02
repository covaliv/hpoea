#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"

#include "hpoea/core/problem.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>
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

constexpr std::string_view kFamily = "ParticleSwarmOptimization";
constexpr std::string_view kImplementation = "pagmo::pso";
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

    ParameterDescriptor omega;
    omega.name = "omega";
    omega.type = ParameterType::Continuous;
    omega.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    omega.default_value = 0.7298;
    space.add_descriptor(omega);

    ParameterDescriptor eta1;
    eta1.name = "eta1";
    eta1.type = ParameterType::Continuous;
    eta1.continuous_range = hpoea::core::ContinuousRange{1.0, 3.0};
    eta1.default_value = 2.05;
    space.add_descriptor(eta1);

    ParameterDescriptor eta2;
    eta2.name = "eta2";
    eta2.type = ParameterType::Continuous;
    eta2.continuous_range = hpoea::core::ContinuousRange{1.0, 3.0};
    eta2.default_value = 2.05;
    space.add_descriptor(eta2);

    ParameterDescriptor max_velocity;
    max_velocity.name = "max_velocity";
    max_velocity.type = ParameterType::Continuous;
    max_velocity.continuous_range = hpoea::core::ContinuousRange{0.0, 100.0};
    max_velocity.default_value = 0.5;
    space.add_descriptor(max_velocity);

    ParameterDescriptor variant;
    variant.name = "variant";
    variant.type = ParameterType::Integer;
    variant.integer_range = hpoea::core::IntegerRange{1, 6};
    variant.default_value = static_cast<std::int64_t>(5);
    space.add_descriptor(variant);

    ParameterDescriptor generations;
    generations.name = "generations";
    generations.type = ParameterType::Integer;
    generations.integer_range = hpoea::core::IntegerRange{1, 1000};
    generations.default_value = static_cast<std::int64_t>(100);
    space.add_descriptor(generations);

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

PagmoParticleSwarmOptimization::PagmoParticleSwarmOptimization()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

PagmoParticleSwarmOptimization::PagmoParticleSwarmOptimization(const PagmoParticleSwarmOptimization &other) = default;

PagmoParticleSwarmOptimization &PagmoParticleSwarmOptimization::operator=(const PagmoParticleSwarmOptimization &other) = default;

void PagmoParticleSwarmOptimization::configure(const ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

OptimizationResult PagmoParticleSwarmOptimization::run(const core::IProblem &problem,
                                                       const core::Budget &budget,
                                                       unsigned long seed) {
    OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        const auto population_size = static_cast<std::size_t>(std::get<std::int64_t>(configured_parameters_.at("population_size")));
        const auto omega = std::get<double>(configured_parameters_.at("omega"));
        const auto eta1 = std::get<double>(configured_parameters_.at("eta1"));
        const auto eta2 = std::get<double>(configured_parameters_.at("eta2"));
        const auto max_velocity = std::get<double>(configured_parameters_.at("max_velocity"));
        const auto variant = static_cast<unsigned>(std::get<std::int64_t>(configured_parameters_.at("variant")));

        auto effective_parameters = configured_parameters_;

        const auto generations = determine_generations(configured_parameters_, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        // pso constructor parameters: gen, omega, eta1, eta2, max_vel, variant, neighb_type, neighb_param, memory, seed
        const auto seed32 = static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());
        pagmo::algorithm algorithm{pagmo::pso(static_cast<unsigned>(generations), omega, eta1, eta2, max_velocity, variant, 2u, 4u, false, seed32)};
        pagmo::problem pg_problem{ProblemAdapter{problem}};

        pagmo::population population{pg_problem, population_size, seed32};

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

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoParticleSwarmOptimization::clone() const {
    return std::make_unique<PagmoParticleSwarmOptimization>(*this);
}

PagmoParticleSwarmOptimizationFactory::PagmoParticleSwarmOptimizationFactory()
    : parameter_space_(make_parameter_space()),
      identity_(make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoParticleSwarmOptimizationFactory::create() const {
    return std::make_unique<PagmoParticleSwarmOptimization>();
}

} // namespace hpoea::pagmo_wrappers

