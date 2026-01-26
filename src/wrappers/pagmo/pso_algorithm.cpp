#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <chrono>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>
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
    d.name = "omega";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.7298;
    space.add_descriptor(d);

    d = {};
    d.name = "eta1";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1.0, 3.0};
    d.default_value = 2.05;
    space.add_descriptor(d);

    d = {};
    d.name = "eta2";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1.0, 3.0};
    d.default_value = 2.05;
    space.add_descriptor(d);

    d = {};
    d.name = "max_velocity";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 100.0};
    d.default_value = 0.5;
    space.add_descriptor(d);

    d = {};
    d.name = "variant";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 6};
    d.default_value = std::int64_t{5};
    space.add_descriptor(d);

    d = {};
    d.name = "generations";
    d.type = ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 1000};
    d.default_value = std::int64_t{100};
    space.add_descriptor(d);

    return space;
}

AlgorithmIdentity make_identity() {
    return {"ParticleSwarmOptimization", "pagmo::pso", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoParticleSwarmOptimization::PagmoParticleSwarmOptimization()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

PagmoParticleSwarmOptimization::PagmoParticleSwarmOptimization(const PagmoParticleSwarmOptimization &other) = default;

PagmoParticleSwarmOptimization &PagmoParticleSwarmOptimization::operator=(const PagmoParticleSwarmOptimization &other) = default;

void PagmoParticleSwarmOptimization::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

OptimizationResult PagmoParticleSwarmOptimization::run(const core::IProblem &problem,
                                                       const core::Budget &budget,
                                                       unsigned long seed) {
    OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        const auto population_size = get_int_param(configured_parameters_, "population_size");
        const auto omega = get_double_param(configured_parameters_, "omega");
        const auto eta1 = get_double_param(configured_parameters_, "eta1");
        const auto eta2 = get_double_param(configured_parameters_, "eta2");
        const auto max_velocity = get_double_param(configured_parameters_, "max_velocity");
        const auto variant = static_cast<unsigned>(get_int_param(configured_parameters_, "variant"));

        auto effective_parameters = configured_parameters_;
        const auto generations = compute_generations(configured_parameters_, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        const auto seed32 = to_seed32(seed);
        pagmo::algorithm algorithm{pagmo::pso(static_cast<unsigned>(generations), omega, eta1, eta2, max_velocity, variant, 2u, 4u, false, seed32)};
        pagmo::problem pg_problem{ProblemAdapter{problem}};
        pagmo::population population{pg_problem, population_size, seed32};

        const auto start_time = std::chrono::steady_clock::now();
        population = algorithm.evolve(population);
        const auto end_time = std::chrono::steady_clock::now();

        result.best_fitness = population.champion_f()[0];
        result.best_solution = population.champion_x();
        result.budget_usage.function_evaluations = population_size * generations;
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

