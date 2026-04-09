#include "test_harness.hpp"

#include "hpoea/core/types.hpp"

#include <chrono>
#include <optional>

using hpoea::core::Budget;
using hpoea::core::EffectiveBudget;

int main() {
    hpoea::tests_v2::TestRunner runner;

    Budget budget;
    budget.function_evaluations = 42u;
    budget.generations = 7u;
    budget.wall_time = std::chrono::milliseconds{1234};

    const EffectiveBudget effective = hpoea::core::to_effective_budget(
        budget,
        std::optional<std::size_t>{10u},
        std::optional<std::size_t>{100u},
        std::optional<std::chrono::milliseconds>{std::chrono::milliseconds{50}});
    HPOEA_V2_CHECK(runner, effective.generations == 10u,
                   "effective_budget.generations override is applied");
    HPOEA_V2_CHECK(runner, effective.function_evaluations == 100u,
                   "effective_budget.function_evaluations override is applied");
    HPOEA_V2_CHECK(runner, effective.wall_time == std::chrono::milliseconds{50},
                   "effective_budget.wall_time override is applied");

    const EffectiveBudget effective_fallback = hpoea::core::to_effective_budget(budget);
    HPOEA_V2_CHECK(runner, effective_fallback.function_evaluations == budget.function_evaluations,
                   "effective_budget falls back to budget.function_evaluations");
    HPOEA_V2_CHECK(runner, effective_fallback.generations == budget.generations,
                   "effective_budget falls back to budget.generations");
    HPOEA_V2_CHECK(runner, effective_fallback.wall_time == budget.wall_time,
                   "effective_budget falls back to budget.wall_time");

    Budget empty_budget;
    const EffectiveBudget empty_effective = hpoea::core::to_effective_budget(empty_budget);
    HPOEA_V2_CHECK(runner, !empty_effective.function_evaluations.has_value(),
                   "empty effective_budget.function_evaluations is nullopt");
    HPOEA_V2_CHECK(runner, !empty_effective.generations.has_value(),
                   "empty effective_budget.generations is nullopt");
    HPOEA_V2_CHECK(runner, !empty_effective.wall_time.has_value(),
                   "empty effective_budget.wall_time is nullopt");

    return runner.summarize("types_budget_tests");
}
