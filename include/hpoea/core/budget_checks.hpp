#pragma once

#include "hpoea/core/types.hpp"

#include <chrono>
#include <cstddef>
#include <string>

namespace hpoea::core {

namespace detail {

// shared by optimizer and algorithm budget checks so messages match
inline void apply_budget_status_counters(const Budget &budget,
                                         std::chrono::milliseconds wall_time,
                                         std::size_t function_evaluations,
                                         std::size_t generations,
                                         RunStatus &status,
                                         std::string &message) {
    if (status != RunStatus::Success &&
        status != RunStatus::BudgetExceeded) {
        return;
    }
    if (budget.wall_time.has_value() && wall_time > *budget.wall_time) {
        status = RunStatus::BudgetExceeded;
        message = "wall-time budget exceeded";
        return;
    }
    if (budget.function_evaluations.has_value() &&
        function_evaluations > *budget.function_evaluations) {
        status = RunStatus::BudgetExceeded;
        message = "function-evaluations budget exceeded";
        return;
    }
    if (budget.generations.has_value() && generations > *budget.generations) {
        status = RunStatus::BudgetExceeded;
        message = "generation budget exceeded";
        return;
    }
}

} // namespace detail

inline void apply_optimizer_budget_status(const Budget &budget,
                                          const OptimizerRunUsage &usage,
                                          RunStatus &status,
                                          std::string &message) {
    detail::apply_budget_status_counters(budget, usage.wall_time,
                                         usage.objective_calls, usage.iterations,
                                         status, message);
}

}
