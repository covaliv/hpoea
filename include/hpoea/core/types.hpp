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

struct BudgetUsage {
    std::size_t function_evaluations{0};
    std::size_t generations{0};
    std::chrono::milliseconds wall_time{0};
};

struct RequestedBudget {
    std::optional<std::size_t> function_evaluations;
    std::optional<std::size_t> generations;
    std::optional<std::chrono::milliseconds> wall_time;
};

struct EffectiveBudget {
    std::optional<std::size_t> function_evaluations;
    std::optional<std::size_t> generations;
    std::optional<std::chrono::milliseconds> wall_time;
};

struct ObservedUsage {
    std::size_t function_evaluations{0};
    std::size_t generations{0};
    std::chrono::milliseconds wall_time{0};
};

[[nodiscard]] inline RequestedBudget to_requested_budget(const Budget &budget) {
    return {
        budget.function_evaluations,
        budget.generations,
        budget.wall_time,
    };
}

[[nodiscard]] inline EffectiveBudget to_effective_budget(const Budget &budget,
                                                         std::optional<std::size_t> generations = std::nullopt,
                                                         std::optional<std::size_t> function_evaluations = std::nullopt,
                                                         std::optional<std::chrono::milliseconds> wall_time = std::nullopt) {
    EffectiveBudget out;
    out.generations = generations.has_value() ? generations : budget.generations;
    out.function_evaluations = function_evaluations.has_value() ? function_evaluations : budget.function_evaluations;
    out.wall_time = wall_time.has_value() ? wall_time : budget.wall_time;
    return out;
}

[[nodiscard]] inline ObservedUsage to_observed_usage(const BudgetUsage &usage) {
    return {
        usage.function_evaluations,
        usage.generations,
        usage.wall_time,
    };
}

[[nodiscard]] inline BudgetUsage legacy_budget_usage(const ObservedUsage &usage) {
    return {
        usage.function_evaluations,
        usage.generations,
        usage.wall_time,
    };
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
