#pragma once

#include "hpoea/core/error_classification.hpp"
#include "hyper_tuning_udp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <pagmo/population.hpp>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace hpoea::pagmo_wrappers {

inline std::shared_ptr<HyperparameterTuningProblem::Context>
make_hyper_context(const core::IEvolutionaryAlgorithmFactory &factory,
                   const core::IProblem &problem,
                   const core::Budget &algorithm_budget,
                   unsigned long seed,
                   std::shared_ptr<core::SearchSpace> search_space = nullptr) {
    if (factory.parameter_space().empty()) {
        throw std::invalid_argument("algorithm has no tunable parameters");
    }
    if (problem.dimension() == 0) {
        throw std::invalid_argument("problem dimension cannot be zero");
    }

    auto ctx = std::make_shared<HyperparameterTuningProblem::Context>();
    ctx->factory = &factory;
    ctx->problem = &problem;
    ctx->algorithm_budget = algorithm_budget;
    ctx->base_seed = seed;
    ctx->trials = std::make_shared<std::vector<core::HyperparameterTrialRecord>>();
    ctx->search_space = std::move(search_space);
    return ctx;
}

// nan-safe comparator for trial records.
// treats non-finite fitness values as worse than any finite value.
inline auto nan_safe_trial_comparator() {
    return [](const core::HyperparameterTrialRecord &a,
              const core::HyperparameterTrialRecord &b) {
        const auto fa = a.optimization_result.best_fitness;
        const auto fb = b.optimization_result.best_fitness;
        if (!std::isfinite(fa)) return false;
        if (!std::isfinite(fb)) return true;
        return fa < fb;
    };
}

inline core::HyperparameterOptimizationResult fill_hyper_result(
    HyperparameterTuningProblem::Context &ctx,
    const pagmo::population &population,
    std::size_t generations,
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end,
    const core::ParameterSet &optimizer_params) {
    core::HyperparameterOptimizationResult result;
    result.status = core::RunStatus::Success;

    if (ctx.trials) {
        result.trials = std::move(*ctx.trials);
    }

    if (auto best = ctx.get_best_trial()) {
        result.best_parameters = best->parameters;
        result.best_objective = best->optimization_result.best_fitness;
    } else if (!result.trials.empty()) {
        auto it = std::min_element(result.trials.begin(), result.trials.end(),
            nan_safe_trial_comparator());
        if (it != result.trials.end() && std::isfinite(it->optimization_result.best_fitness)) {
            result.best_parameters = it->parameters;
            result.best_objective = it->optimization_result.best_fitness;
        } else {
            result.status = core::RunStatus::InternalError;
            result.message = "no valid hyperparameter trial was recorded";
            result.best_objective = std::numeric_limits<double>::infinity();
        }
    } else {
        result.status = core::RunStatus::InternalError;
        result.message = "no valid hyperparameter trial was recorded";
        result.best_objective = std::numeric_limits<double>::infinity();
    }

    result.optimizer_usage.wall_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    result.optimizer_usage.iterations = generations;
    result.optimizer_usage.objective_calls = ctx.get_evaluations();
    result.effective_optimizer_parameters = optimizer_params;
    if (result.status == core::RunStatus::Success) {
        result.message = "hyperparameter optimization completed";
    }
    return result;
}

// main entry point for running a hyper-optimization loop.
//
// AlgorithmSetup is a callable:
//   (pagmo::problem &tuning_problem, const std::pair<pagmo::vector_double, pagmo::vector_double> &bounds,
//    const core::Budget &optimizer_budget, std::chrono::steady_clock::time_point start,
//    HyperparameterTuningProblem::Context &ctx)
//   -> std::pair<pagmo::population, std::size_t>
//
// it should create/evolve a population and return it along with actual_iterations.
template <typename AlgorithmSetup>
core::HyperparameterOptimizationResult run_hyper_optimization(
    const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    const core::IProblem &problem,
    const core::Budget &optimizer_budget,
    const core::Budget &algorithm_budget,
    unsigned long seed,
    const core::ParameterSet &configured_parameters,
    const std::shared_ptr<core::SearchSpace> &search_space,
    AlgorithmSetup &&setup) {

    static_assert(
        std::is_invocable_r_v<
            std::pair<pagmo::population, std::size_t>,
            AlgorithmSetup,
            pagmo::problem &,
            const std::pair<pagmo::vector_double, pagmo::vector_double> &,
            const core::Budget &,
            std::chrono::steady_clock::time_point,
            HyperparameterTuningProblem::Context &>,
        "AlgorithmSetup must be callable with "
        "(problem&, bounds&, budget&, start_time, context&) -> pair<population, size_t>");

    core::HyperparameterOptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    std::shared_ptr<HyperparameterTuningProblem::Context> ctx;
    const auto start_time = std::chrono::steady_clock::now();

    try {
        if (search_space) {
            search_space->validate(algorithm_factory.parameter_space());
        }

        ctx = make_hyper_context(algorithm_factory, problem, algorithm_budget, seed, search_space);
        HyperparameterTuningProblem udp{ctx};

        const auto bounds = udp.get_bounds();
        pagmo::problem tuning_problem{udp};

        auto [population, actual_iterations] =
            setup(tuning_problem, bounds, optimizer_budget, start_time, *ctx);

        const auto end_time = std::chrono::steady_clock::now();

        result = fill_hyper_result(*ctx, population, actual_iterations,
                                   start_time, end_time, configured_parameters);
        result.seed = seed;
        apply_optimizer_budget_status(
            optimizer_budget,
            result.optimizer_usage,
            result.status,
            result.message);
    } catch (const std::exception &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.optimizer_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.message = ex.what();

        const auto classified = core::classify_exception(ex);
        result.status = classified.status;
        result.error_info = classified.error_info;
    }

    // fallback: if we have trials from before the exception, recover the best one
    if (ctx && ctx->trials && !ctx->trials->empty() && result.trials.empty()) {
        result.trials = std::move(*ctx->trials);
        auto it = std::min_element(result.trials.begin(), result.trials.end(),
            nan_safe_trial_comparator());
        if (it != result.trials.end() && std::isfinite(it->optimization_result.best_fitness)) {
            result.best_parameters = it->parameters;
            result.best_objective = it->optimization_result.best_fitness;
        }
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers
