#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>

namespace hpoea::core {

enum class RunStatus {
    Success,
    BudgetExceeded,
    FailedEvaluation,
    InvalidConfiguration,
    InternalError
};

struct Budget {
    std::optional<std::size_t> function_evaluations;
    std::optional<std::size_t> generations;
    std::optional<std::chrono::milliseconds> wall_time;
};

// usage reported by a single ea run (inner data path).
struct AlgorithmRunUsage {
    std::size_t function_evaluations{0};
    std::size_t generations{0};
    std::chrono::milliseconds wall_time{0};
};

// usage counters for the outer hyperparameter optimizer.
struct OptimizerRunUsage {
    std::size_t objective_calls{0};  // number of complete ea runs
    std::size_t iterations{0};       // optimizer-side stepping
    std::chrono::milliseconds wall_time{0};
};

using EffectiveBudget = Budget;

[[nodiscard]] inline EffectiveBudget to_effective_budget(const Budget &budget,
                                                         std::optional<std::size_t> generations = std::nullopt,
                                                         std::optional<std::size_t> function_evaluations = std::nullopt,
                                                         std::optional<std::chrono::milliseconds> wall_time = std::nullopt) {
    EffectiveBudget out;
    out.generations = generations ? generations : budget.generations;
    out.function_evaluations = function_evaluations ? function_evaluations : budget.function_evaluations;
    out.wall_time = wall_time ? wall_time : budget.wall_time;
    return out;
}

struct AlgorithmIdentity {
    std::string family;
    std::string implementation;
    std::string version;
};

struct ErrorInfo {
    std::string category;
    std::string code;
    std::string detail;
};

} // namespace hpoea::core
