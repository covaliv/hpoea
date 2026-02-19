#include "hpoea/wrappers/pagmo/cmaes_algorithm.hpp"

#include "budget_util.hpp"
#include "problem_adapter.hpp"

#include <chrono>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/cmaes.hpp>
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
    d.name = "sigma0";
    d.type = ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{1e-6, 5.0};
    d.default_value = 0.5;
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
    return {"CMAES", "pagmo::cmaes", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoCmaes::PagmoCmaes()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoCmaes::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

core::OptimizationResult PagmoCmaes::run(const core::IProblem &problem,
                                         const core::Budget &budget,
                                         unsigned long seed) {
    OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    try {
        const auto population_size = get_int_param(configured_parameters_, "population_size");
        const auto sigma0 = get_double_param(configured_parameters_, "sigma0");
        const auto ftol = get_double_param(configured_parameters_, "ftol");
        const auto xtol = get_double_param(configured_parameters_, "xtol");

        auto effective_parameters = configured_parameters_;
        const auto generations = compute_generations(configured_parameters_, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        const auto algo_seed = to_seed32(seed);
        const auto pop_seed = derive_seed(seed, 1);
        pagmo::algorithm algorithm{pagmo::cmaes(static_cast<unsigned>(generations), -1, -1, -1, -1, sigma0, ftol, xtol, true, false, algo_seed)};
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

std::unique_ptr<core::IEvolutionaryAlgorithm> PagmoCmaes::clone() const {
    return std::make_unique<PagmoCmaes>(*this);
}

PagmoCmaesFactory::PagmoCmaesFactory()
    : parameter_space_(make_parameter_space()),
      identity_(make_identity()) {}

core::EvolutionaryAlgorithmPtr PagmoCmaesFactory::create() const {
    return std::make_unique<PagmoCmaes>();
}

} // namespace hpoea::pagmo_wrappers


