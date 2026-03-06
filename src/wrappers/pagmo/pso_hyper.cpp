#include "hpoea/wrappers/pagmo/pso_hyper.hpp"

#include "budget_util.hpp"
#include "hyper_tuning_udp.hpp"
#include "hyper_util.hpp"

#include <algorithm>
#include <chrono>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;

ParameterSpace make_parameter_space() {
    ParameterSpace space;

    ParameterDescriptor d;
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

    return space;
}

AlgorithmIdentity make_identity() {
    return {"PSO", "pagmo::pso", "2.x"};
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoPsoHyperOptimizer::PagmoPsoHyperOptimizer()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

core::HyperparameterOptimizerPtr PagmoPsoHyperOptimizer::clone() const {
    return std::make_unique<PagmoPsoHyperOptimizer>(*this);
}

void PagmoPsoHyperOptimizer::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

void PagmoPsoHyperOptimizer::set_search_space(
    std::shared_ptr<core::SearchSpace> search_space) {
    search_space_ = std::move(search_space);
}

core::HyperparameterOptimizationResult PagmoPsoHyperOptimizer::optimize(
    const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    const core::IProblem &problem,
    const core::Budget &optimizer_budget,
    const core::Budget &algorithm_budget,
    unsigned long seed) {

    core::HyperparameterOptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    std::shared_ptr<HyperparameterTuningProblem::Context> ctx;
    const auto start_time = std::chrono::steady_clock::now();

    try {
        if (search_space_) {
            search_space_->validate(algorithm_factory.parameter_space());
        }

        ctx = make_hyper_context(algorithm_factory, problem, algorithm_budget, seed, search_space_);
        HyperparameterTuningProblem udp{ctx};

        const auto bounds = udp.get_bounds();
        pagmo::problem tuning_problem{udp};

        const auto omega = std::get<double>(configured_parameters_.at("omega"));
        const auto eta1 = std::get<double>(configured_parameters_.at("eta1"));
        const auto eta2 = std::get<double>(configured_parameters_.at("eta2"));
        const auto max_vel = std::get<double>(configured_parameters_.at("max_velocity"));
        const auto variant = static_cast<unsigned>(
            std::get<std::int64_t>(configured_parameters_.at("variant")));

        const auto seed32 = to_seed32(seed);
        const auto dim = bounds.first.size();
        const auto pop_size = static_cast<pagmo::population::size_type>(std::max(dim * 4, dim + 1));
        const auto generations = static_cast<unsigned>(
            compute_generations(configured_parameters_, optimizer_budget, static_cast<std::size_t>(pop_size)));

        pagmo::algorithm algorithm{
            pagmo::pso(generations, omega, eta1, eta2, max_vel, variant, 2u, 4u, false, seed32)};

        const auto t0 = std::chrono::steady_clock::now();
        pagmo::population population{tuning_problem, pop_size, derive_seed(seed, 1)};
        if (generations > 0) {
            population = algorithm.evolve(population);
        }
        const auto t1 = std::chrono::steady_clock::now();

        fill_hyper_result(result, *ctx, population, generations, t0, t1, configured_parameters_);
        apply_budget_status(
            optimizer_budget,
            result.budget_usage,
            result.status,
            result.message);
    } catch (const core::ParameterValidationError &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.budget_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.status = core::RunStatus::InvalidConfiguration;
        result.message = ex.what();
    } catch (const std::invalid_argument &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.budget_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.status = core::RunStatus::InvalidConfiguration;
        result.message = ex.what();
    } catch (const std::exception &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.budget_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.status = core::RunStatus::InternalError;
        result.message = ex.what();
    }

    if (ctx && ctx->trials && !ctx->trials->empty() && result.trials.empty()) {
        result.trials = std::move(*ctx->trials);
        auto it = std::min_element(result.trials.begin(), result.trials.end(),
            [](const auto &a, const auto &b) {
                return a.optimization_result.best_fitness < b.optimization_result.best_fitness;
            });
        result.best_parameters = it->parameters;
        result.best_objective = it->optimization_result.best_fitness;
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers
