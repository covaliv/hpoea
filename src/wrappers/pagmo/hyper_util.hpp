#pragma once

#include "hyper_tuning_udp.hpp"

#include <chrono>
#include <limits>
#include <memory>
#include <pagmo/population.hpp>
#include <stdexcept>
#include <vector>

namespace hpoea::pagmo_wrappers {

// creates context for hyperparameter optimization.
// IMPORTANT: factory and problem must remain valid for the lifetime of the
// returned context. the caller is responsible for ensuring this.
inline std::shared_ptr<HyperTuningUdp::Context>
make_hyper_context(const core::IEvolutionaryAlgorithmFactory &factory,
                   const core::IProblem &problem,
                   const core::Budget &budget,
                   unsigned long seed,
                   std::shared_ptr<core::SearchSpace> search_space = nullptr) {
    if (factory.parameter_space().empty()) {
        throw std::invalid_argument("algorithm has no tunable parameters");
    }
    if (problem.dimension() == 0) {
        throw std::invalid_argument("problem dimension cannot be zero");
    }

    auto ctx = std::make_shared<HyperTuningUdp::Context>();
    ctx->factory = &factory;
    ctx->problem = &problem;
    ctx->algorithm_budget = budget;
    ctx->base_seed = seed;
    ctx->trials = std::make_shared<std::vector<core::HyperparameterTrialRecord>>();
    ctx->search_space = std::move(search_space);
    return ctx;
}

inline void fill_hyper_result(core::HyperparameterOptimizationResult &result,
                              HyperTuningUdp::Context &ctx,
                              const pagmo::population &population,
                              std::size_t generations,
                              std::chrono::steady_clock::time_point start,
                              std::chrono::steady_clock::time_point end,
                              const core::ParameterSet &optimizer_params) {
    result.status = core::RunStatus::Success;

    if (ctx.trials) {
        result.trials = std::move(*ctx.trials);
    }

    if (auto best = ctx.get_best_trial()) {
        result.best_parameters = best->parameters;
        result.best_objective = best->optimization_result.best_fitness;
    } else if (!result.trials.empty()) {
        // fallback: find best trial from recorded results
        auto it = std::min_element(result.trials.begin(), result.trials.end(),
            [](const auto &a, const auto &b) {
                return a.optimization_result.best_fitness < b.optimization_result.best_fitness;
            });
        result.best_parameters = it->parameters;
        result.best_objective = it->optimization_result.best_fitness;
    } else {
        const auto &f = population.champion_f();
        result.best_objective = f.empty() ? std::numeric_limits<double>::max() : f[0];
    }

    result.budget_usage.wall_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    result.budget_usage.generations = generations;
    result.budget_usage.function_evaluations = ctx.get_evaluations();
    result.effective_optimizer_parameters = optimizer_params;
    result.message = "hyperparameter optimization completed";
}

} // namespace hpoea::pagmo_wrappers
