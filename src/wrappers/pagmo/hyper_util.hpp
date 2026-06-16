#pragma once

#include "hpoea/core/budget_checks.hpp"
#include "hpoea/core/error_classification.hpp"
#include "hyper_tuning_udp.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <pagmo/population.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

// outcome of an optimizer's evolve lambda
// consumed by fill_hyper_result
struct HyperEvolveOutcome {
    std::size_t iterations{0};
    core::ParameterSet effective_parameters;  // configured params with control value clamped to budget
    std::string starved_message;  // non-empty when budget too small for any evolve so not Success
};

inline core::HyperparameterOptimizationResult fill_hyper_result(
    HyperparameterTuningProblem::Context &ctx,
    HyperEvolveOutcome outcome,
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
    core::HyperparameterOptimizationResult result;

    if (ctx.trials) {
        result.trials = std::move(*ctx.trials);
    }

    // never report Success on an all-failed run
    // mirrors random_search_optimizer.cpp
    if (auto best = ctx.get_best_trial(); best && is_selectable_trial(*best)) {
        result.status = core::RunStatus::Success;
        result.best_parameters = best->parameters;
        result.best_objective = best->optimization_result.best_fitness;
        result.message = "hyperparameter optimization completed";
    } else if (!result.trials.empty()) {
        const auto &first = result.trials.front().optimization_result;
        result.status = first.status;
        result.best_objective = std::numeric_limits<double>::infinity();
        result.error_info = first.error_info;
        result.message = first.message.empty()
            ? "hyperparameter optimization produced no successful finite trial"
            : first.message;
    } else {
        result.status = core::RunStatus::InternalError;
        result.best_objective = std::numeric_limits<double>::infinity();
        result.error_info = core::ErrorInfo{
            "internal_error", "no_valid_trial", "hyperparameter optimization produced no trial"};
        result.message = "no valid hyperparameter trial was recorded";
    }

    // zero outer iterations did no real work despite initial pop
    // not Success
    if (!outcome.starved_message.empty() && result.status == core::RunStatus::Success) {
        result.status = core::RunStatus::BudgetExceeded;
        result.message = std::move(outcome.starved_message);
    }

    result.optimizer_usage.wall_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    result.optimizer_usage.iterations = outcome.iterations;
    result.optimizer_usage.objective_calls = ctx.get_evaluations();
    result.effective_optimizer_parameters = std::move(outcome.effective_parameters);
    return result;
}

// main entry point for running a hyper-optimization loop.
//
// AlgorithmSetup is a callable:
//   (pagmo::problem &tuning_problem, const std::pair<pagmo::vector_double, pagmo::vector_double> &bounds,
//    const core::Budget &optimizer_budget, std::chrono::steady_clock::time_point start,
//    HyperparameterTuningProblem::Context &ctx)
//   -> HyperEvolveOutcome
template <typename AlgorithmSetup>
core::HyperparameterOptimizationResult run_hyper_optimization(
    const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    const core::IProblem &problem,
    const core::Budget &optimizer_budget,
    const core::Budget &algorithm_budget,
    unsigned long seed,
    const std::shared_ptr<core::SearchSpace> &search_space,
    AlgorithmSetup &&setup) {

    static_assert(
        std::is_invocable_r_v<
            HyperEvolveOutcome,
            AlgorithmSetup,
            pagmo::problem &,
            const std::pair<pagmo::vector_double, pagmo::vector_double> &,
            const core::Budget &,
            std::chrono::steady_clock::time_point,
            HyperparameterTuningProblem::Context &>,
        "AlgorithmSetup must be callable with "
        "(problem&, bounds&, budget&, start_time, context&) -> HyperEvolveOutcome");

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

        auto outcome = setup(tuning_problem, bounds, optimizer_budget, start_time, *ctx);

        const auto end_time = std::chrono::steady_clock::now();

        result = fill_hyper_result(*ctx, std::move(outcome), start_time, end_time);
        result.seed = seed;
        core::apply_optimizer_budget_status(
            optimizer_budget,
            result.optimizer_usage,
            result.status,
            result.message);
    } catch (const std::exception &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.optimizer_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // recover evaluations done before the throw so a crashed run still counts them
        if (ctx) {
            result.optimizer_usage.objective_calls = ctx->get_evaluations();
        }
        result.message = ex.what();

        const auto classified = core::classify_exception(ex);
        result.status = classified.status;
        result.error_info = classified.error_info;
    }

    // fallback path
    // recover trials from before the exception
    // pick the best selectable one
    if (ctx && !ctx->trials->empty() && result.trials.empty()) {
        result.trials = std::move(*ctx->trials);
        if (auto best = ctx->get_best_trial(); best && is_selectable_trial(*best)) {
            result.best_parameters = best->parameters;
            result.best_objective = best->optimization_result.best_fitness;
        }
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers
