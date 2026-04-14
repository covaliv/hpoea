#include "test_harness.hpp"

#include "hpoea/config/config_validator.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::ExperimentSpec;
using hpoea::config::OptimizerSpec;
using hpoea::config::ProblemSpec;
using hpoea::config::SuiteConfig;
using hpoea::config::ValidationResult;

bool has_error_path(const ValidationResult &result,
                    const std::string &path) {
    for (const auto &diag : result.diagnostics) {
        if (diag.path == path) {
            return true;
        }
    }
    return false;
}

SuiteConfig make_valid_suite() {
    SuiteConfig cfg;
    cfg.schema_version = 1;
    cfg.name = "valid_suite";
    cfg.output_dir = "results/valid_suite";
    cfg.repetitions = 2;

    ProblemSpec problem;
    problem.id = "sphere10";
    problem.type = "sphere";
    cfg.problems.push_back(std::move(problem));

    AlgorithmSpec algorithm;
    algorithm.id = "ea_default";
    algorithm.type = "research_ea";
    cfg.algorithms.push_back(std::move(algorithm));

    OptimizerSpec optimizer;
    optimizer.id = "optimizer_fast";
    optimizer.type = "research_optimizer";
    cfg.optimizers.push_back(std::move(optimizer));

    ExperimentSpec experiment;
    experiment.id = "sphere_ea";
    experiment.problem = "sphere10";
    experiment.algorithm = "ea_default";
    experiment.optimizer = "optimizer_fast";
    cfg.experiments.push_back(std::move(experiment));

    return cfg;
}

} // namespace

int main() {
    hpoea::tests_v2::TestRunner runner;

    {
        const auto result = hpoea::config::validate_suite_config(make_valid_suite());
        HPOEA_V2_CHECK(runner, result.ok(), "valid explicit config validates");
    }

    {
        auto cfg = make_valid_suite();
        cfg.problems.push_back(cfg.problems.front());
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "problems.sphere10"), "duplicate problem ids fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().problem = "missing_problem";
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].problem"), "unknown problem refs fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().repetitions = 0;
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].repetitions"),
                       "invalid experiment repetitions fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().optimizer_budget = hpoea::config::BudgetConfig{std::nullopt, 0};
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].optimizer_budget.function_evaluations"),
                       "invalid budgets fail");
    }

    return runner.summarize("config_validator_tests");
}
