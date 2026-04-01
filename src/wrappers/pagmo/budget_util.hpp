#pragma once

#include "hpoea/core/error_classification.hpp"
#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"
#include "problem_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
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
    return static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());
}

// derive a different seed by mixing with a salt value.
inline unsigned derive_seed(unsigned long seed, unsigned long salt) {
    std::mt19937 rng(to_seed32(seed ^ (salt * 2654435761UL)));
    return rng();
}

inline std::size_t read_fevals(const pagmo::population &population,
                               std::size_t fallback) {
    try {
        return static_cast<std::size_t>(population.get_problem().get_fevals());
    } catch (const std::exception &) {
        // pagmo may not track fevals for all problem types;
        // fall back to the population-based estimate
        return fallback;
    }
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

    if (budget.generations)
        gens = std::min(gens, *budget.generations);

    if (budget.function_evaluations) {
        // reserve population_size fevals for initial population
        auto available = *budget.function_evaluations > population_size
                             ? *budget.function_evaluations - population_size
                             : 0;
        auto max_gens = available / population_size;
        gens = std::min(gens, max_gens);
    }

    return gens;
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
        auto available = *budget.function_evaluations > ps
            ? *budget.function_evaluations - ps : std::size_t{0};
        generations = std::min(generations, available / ps);
    }
    return generations;
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

// check whether an inner ea run exceeded its budget.
inline void apply_budget_status(const core::Budget &budget,
                                const core::AlgorithmRunUsage &usage,
                                core::RunStatus &status,
                                std::string &message) {
    if (status != core::RunStatus::Success &&
        status != core::RunStatus::BudgetExceeded) {
        return;
    }

    if (budget.wall_time.has_value() && usage.wall_time > *budget.wall_time) {
        status = core::RunStatus::BudgetExceeded;
        message = "wall-time budget exceeded";
        return;
    }
    if (budget.function_evaluations.has_value() &&
        usage.function_evaluations > *budget.function_evaluations) {
        status = core::RunStatus::BudgetExceeded;
        message = "function-evaluations budget exceeded";
        return;
    }
    if (budget.generations.has_value() && usage.generations > *budget.generations) {
        status = core::RunStatus::BudgetExceeded;
        message = "generation budget exceeded";
        return;
    }
}

// check whether an outer hyper-optimizer run exceeded its budget.
inline void apply_optimizer_budget_status(const core::Budget &budget,
                                          const core::OptimizerRunUsage &usage,
                                          core::RunStatus &status,
                                          std::string &message) {
    if (status != core::RunStatus::Success &&
        status != core::RunStatus::BudgetExceeded) {
        return;
    }
    if (budget.wall_time.has_value() && usage.wall_time > *budget.wall_time) {
        status = core::RunStatus::BudgetExceeded;
        message = "wall-time budget exceeded";
        return;
    }
    if (budget.function_evaluations.has_value() &&
        usage.objective_calls > *budget.function_evaluations) {
        status = core::RunStatus::BudgetExceeded;
        message = "function-evaluations budget exceeded";
        return;
    }
    if (budget.generations.has_value() && usage.iterations > *budget.generations) {
        status = core::RunStatus::BudgetExceeded;
        message = "generation budget exceeded";
        return;
    }
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

    try {
        static_assert(std::is_invocable_r_v<pagmo::algorithm,
                                             AlgorithmBuilder,
                                             unsigned,
                                             unsigned>,
                      "AlgorithmBuilder must be callable as pagmo::algorithm(unsigned generations, unsigned seed32)");

        const auto population_size = get_param<std::int64_t>(configured_parameters, "population_size");

        auto effective_parameters = configured_parameters;
        const auto generations = compute_generations(configured_parameters, budget, population_size);
        effective_parameters.insert_or_assign("generations", static_cast<std::int64_t>(generations));

        const auto algo_seed = to_seed32(seed);
        const auto pop_seed = derive_seed(seed, 1);
        pagmo::algorithm algorithm = make_algorithm(static_cast<unsigned>(generations), algo_seed);
        pagmo::problem pg_problem{ProblemAdapter{problem}};
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
        // back-derive actual generations from actual fevals.
        // all wrapped algorithms evaluate exactly population_size individuals per generation.
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

        result.status = core::RunStatus::Success;
        result.message = "optimization completed";
        apply_budget_status(budget, result.algorithm_usage, result.status, result.message);
    } catch (const std::exception &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.algorithm_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.message = ex.what();

        const auto classified = core::classify_exception(ex);
        result.status = classified.status;
        result.error_info = classified.error_info;
    }

    return result;
}

} // namespace hpoea::pagmo_wrappers
