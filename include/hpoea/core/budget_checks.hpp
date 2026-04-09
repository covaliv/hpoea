#pragma once

#include "hpoea/core/types.hpp"

#include <string>

namespace hpoea::core {

inline void apply_optimizer_budget_status(const Budget &budget,
                                          const OptimizerRunUsage &usage,
                                          RunStatus &status,
                                          std::string &message) {
    if (status != RunStatus::Success &&
        status != RunStatus::BudgetExceeded) {
        return;
    }
    if (budget.wall_time.has_value() && usage.wall_time > *budget.wall_time) {
        status = RunStatus::BudgetExceeded;
        message = "wall-time budget exceeded";
        return;
    }
    if (budget.function_evaluations.has_value() &&
        usage.objective_calls > *budget.function_evaluations) {
        status = RunStatus::BudgetExceeded;
        message = "function-evaluations budget exceeded";
        return;
    }
    if (budget.generations.has_value() && usage.iterations > *budget.generations) {
        status = RunStatus::BudgetExceeded;
        message = "generation budget exceeded";
        return;
    }
}

}
