#pragma once

#include "hpoea/core/budget_checks.hpp"
#include "hpoea/core/error_classification.hpp"
#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/seeding.hpp"
#include "hpoea/core/types.hpp"
#include "problem_adapter.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <pagmo/algorithm.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

namespace hpoea::pagmo_wrappers {

template <typename T>
inline auto get_param(const core::ParameterSet &params, const char *name) {
    auto it = params.find(name);
    if (it == params.end()) {
        throw std::invalid_argument(std::string("missing parameter: ") + name);
    }
    if (!std::holds_alternative<T>(it->second)) {
        throw std::invalid_argument(std::string("parameter '") + name + "' type mismatch");
    }
    if constexpr (std::is_same_v<T, std::int64_t>) {
        auto val = std::get<std::int64_t>(it->second);
        if (val < 0) {
            throw std::invalid_argument(std::string("parameter '") + name + "' cannot be negative");
        }
        return static_cast<std::size_t>(val);
    } else {
        return std::get<T>(it->second);
    }
}

inline unsigned to_seed32(unsigned long seed) {
    return static_cast<unsigned>(core::splitmix64(seed));
}

inline unsigned derive_seed32(unsigned long seed, unsigned long index) {
    return static_cast<unsigned>(core::derive_stream_seed(seed, index));
}

inline std::size_t read_fevals(const pagmo::population &population,
                               std::size_t fallback) {
    try {
        return static_cast<std::size_t>(population.get_problem().get_fevals());
    } catch (const std::exception &) {
        // pagmo may not track fevals for all problem types
        // fall back to population estimate
        return fallback;
    }
}

inline std::size_t clamp_hyper_generations(
    std::size_t configured_generations,
    const core::Budget &budget,
    std::size_t population_size) {
    const auto ps = std::max(population_size, std::size_t{1});
    auto generations = configured_generations;
    if (budget.generations) {
        generations = std::min(generations, *budget.generations);
    }
    if (budget.function_evaluations) {
        // reserve population_size fevals for initial population
        auto available = *budget.function_evaluations > ps
            ? *budget.function_evaluations - ps : std::size_t{0};
        generations = std::min(generations, available / ps);
    }
    return generations;
}

inline std::size_t compute_generations(const core::ParameterSet &params,
                                       const core::Budget &budget,
                                       std::size_t population_size) {
    if (population_size == 0) {
        throw std::invalid_argument("population_size cannot be zero");
    }

    auto gens = get_param<std::int64_t>(params, "generations");
    if (gens == 0) {
        throw std::invalid_argument("generations must be positive");
    }

    return clamp_hyper_generations(gens, budget, population_size);
}

struct BudgetFields {
    core::Budget requested_budget;
    core::EffectiveBudget effective_budget;
};

inline BudgetFields compute_budget_fields(const core::Budget &budget,
                                           std::size_t generations,
                                           std::size_t population_size) {
    BudgetFields fields;
    fields.requested_budget = budget;

    std::optional<std::size_t> effective_fevals;
    if (budget.function_evaluations.has_value()) {
        const auto max_val = std::numeric_limits<std::size_t>::max();
        std::size_t estimated_fevals = max_val;
        if (generations <= (max_val - population_size) / population_size) {
            estimated_fevals = population_size + (generations * population_size);
        }
        effective_fevals = std::min(*budget.function_evaluations, estimated_fevals);
    }

    fields.effective_budget = core::to_effective_budget(
        budget,
        generations,
        effective_fevals,
        budget.wall_time);

    return fields;
}

inline void apply_budget_status(const core::Budget &budget,
                                const core::AlgorithmRunUsage &usage,
                                core::RunStatus &status,
                                std::string &message) {
    core::detail::apply_budget_status_counters(budget, usage.wall_time,
                                               usage.function_evaluations, usage.generations,
                                               status, message);
}

template <typename AlgorithmBuilder>
inline core::OptimizationResult run_population(
    const core::IProblem &problem,
    const core::Budget &budget,
    const core::ParameterSet &configured_parameters,
    unsigned long seed,
    AlgorithmBuilder &&make_algorithm) {
    core::OptimizationResult result;
    result.status = core::RunStatus::InternalError;
    result.seed = seed;

    const auto start_time = std::chrono::steady_clock::now();
    // shared across pagmo's problem copies
    // so the catch path can still recover the fevals
    auto eval_counter = std::make_shared<std::atomic<std::size_t>>(0);
    std::size_t population_size = 0;

    try {
        static_assert(std::is_invocable_r_v<pagmo::algorithm,
                                             AlgorithmBuilder,
                                             unsigned,
                                             unsigned>,
                      "AlgorithmBuilder must be callable as pagmo::algorithm(unsigned generations, unsigned seed32)");

        population_size = get_param<std::int64_t>(configured_parameters, "population_size");

        auto effective_parameters = configured_parameters;
        const auto generations = compute_generations(configured_parameters, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        const auto algo_seed = to_seed32(seed);
        const auto pop_seed = derive_seed32(seed, 0);
        constexpr auto uint_max = static_cast<std::size_t>(std::numeric_limits<unsigned>::max());
        pagmo::algorithm algorithm = make_algorithm(
            static_cast<unsigned>(std::min(generations, uint_max)), algo_seed);
        pagmo::problem pg_problem{ProblemAdapter{problem, eval_counter}};
        pagmo::population population{pg_problem, population_size, pop_seed};

        if (generations > 0) {
            population = algorithm.evolve(population);
        }
        const auto end_time = std::chrono::steady_clock::now();

        const auto &champion_f = population.champion_f();
        if (champion_f.empty()) {
            throw std::runtime_error("pagmo population has empty champion fitness");
        }

        result.best_fitness = champion_f[0];
        result.best_solution = population.champion_x();
        result.algorithm_usage.function_evaluations = read_fevals(
            population,
            population_size * (generations + 1));
        // back-derive generations from fevals
        // every wrapped algorithm does exactly population_size evals per generation
        const auto actual_fevals = result.algorithm_usage.function_evaluations;
        const auto actual_generations = actual_fevals > population_size
            ? (actual_fevals - population_size) / population_size
            : 0u;
        result.algorithm_usage.generations = actual_generations;
        result.algorithm_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto budget_fields = compute_budget_fields(budget, actual_generations, population_size);
        result.requested_budget = budget_fields.requested_budget;
        result.effective_budget = budget_fields.effective_budget;
        result.effective_parameters = std::move(effective_parameters);

        if (generations == 0) {
            result.status = core::RunStatus::BudgetExceeded;
            result.message = "budget insufficient for any generations; only initial population evaluated";
        } else {
            result.status = core::RunStatus::Success;
            result.message = "optimization completed";
            apply_budget_status(budget, result.algorithm_usage, result.status, result.message);
        }
    } catch (const std::exception &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.algorithm_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        // population is gone on throw
        // recover fevals from the shared counter
        // back-derive generations from that
        const auto performed = eval_counter->load(std::memory_order_relaxed);
        result.algorithm_usage.function_evaluations = performed;
        result.algorithm_usage.generations =
            (population_size > 0 && performed > population_size)
                ? (performed - population_size) / population_size
                : 0u;
        result.message = ex.what();

        const auto classified = core::classify_exception(ex);
        result.status = classified.status;
        result.error_info = classified.error_info;
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers
