#include "test_harness.hpp"

#include "hpoea/config/suite_expander.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::ExperimentSpec;
using hpoea::config::OptimizerSpec;
using hpoea::config::ProblemSpec;
using hpoea::config::SuiteConfig;

SuiteConfig make_valid_suite() {
    SuiteConfig cfg;
    cfg.schema_version = 1;
    cfg.name = "valid_suite";
    cfg.output_dir = "results/valid_suite";
    cfg.suite_seed = 4242ULL;
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
        auto cfg = make_valid_suite();
        cfg.experiments.front().repetitions = 3;
        cfg.experiments.front().seed = 9001ULL;
        cfg.experiments.front().output_name = "sphere_ea";
        cfg.experiments.front().algorithm_budget = hpoea::config::BudgetConfig{25, std::nullopt};
        cfg.experiments.front().optimizer_budget = hpoea::config::BudgetConfig{std::nullopt, 800};
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, result.ok(), "valid explicit config expands");
        HPOEA_V2_CHECK(runner, result.runs.size() == 3, "experiment repetitions expand to run count");
        HPOEA_V2_CHECK(runner, result.runs[0].run_id == "sphere_ea__rep000", "run id is deterministic");
        HPOEA_V2_CHECK(runner, result.runs[1].seed == 9002ULL, "explicit seed increments by repetition");
        HPOEA_V2_CHECK(runner, result.runs[0].planned_output_path
                                   == std::filesystem::path{"results/valid_suite/experiments/sphere_ea/run-000.jsonl"},
                       "planned output path is deterministic");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().output_name = "../escape";
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, !result.ok(), "unsafe output names fail expansion");
        HPOEA_V2_CHECK(runner, result.runs.empty(), "failed expansion returns no runs");
    }

    return runner.summarize("suite_expander_tests");
}
