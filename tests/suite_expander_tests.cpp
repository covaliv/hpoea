#include "test_harness.hpp"

#include "hpoea/config/config_parser.hpp"
#include "hpoea/config/suite_expander.hpp"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::ExperimentSpec;
using hpoea::config::ExpansionResult;
using hpoea::config::OptimizerSpec;
using hpoea::config::ProblemSpec;
using hpoea::config::ResolvedRunSpec;
using hpoea::config::SuiteConfig;

std::filesystem::path example_path(const std::string &file_name) {
    return std::filesystem::path{HPOEA_PROJECT_SOURCE_DIR} / "examples" / "configs" / file_name;
}

bool has_error_path(const ExpansionResult &result,
                    const std::string &path) {
    for (const auto &diag : result.diagnostics) {
        if (diag.path == path) {
            return true;
        }
    }
    return false;
}

const ResolvedRunSpec *find_run(const ExpansionResult &result,
                                const std::string &run_id) {
    for (const auto &run : result.runs) {
        if (run.run_id == run_id) {
            return &run;
        }
    }
    return nullptr;
}

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
    problem.parameters.emplace("dimension", std::int64_t{10});
    cfg.problems.push_back(std::move(problem));

    AlgorithmSpec algorithm;
    algorithm.id = "ea_default";
    algorithm.type = "research_ea";
    algorithm.fixed_parameters.emplace("population_size", std::int64_t{50});
    cfg.algorithms.push_back(std::move(algorithm));

    OptimizerSpec optimizer;
    optimizer.id = "optimizer_fast";
    optimizer.type = "research_optimizer";
    optimizer.parameters.emplace("generations", std::int64_t{10});
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
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, result.ok(), "valid basic config expands");
        HPOEA_V2_CHECK(runner, result.runs.size() == 3,
                       "experiment repetitions expand to run count");
        HPOEA_V2_CHECK(runner, find_run(result, "sphere_ea__rep000") != nullptr,
                       "run ids start at repetition zero");
        HPOEA_V2_CHECK(runner, find_run(result, "sphere_ea__rep002") != nullptr,
                       "run ids are deterministic");
    }

    {
        auto cfg = make_valid_suite();
        cfg.repetitions = 4;
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, result.runs.size() == 4,
                       "suite repetitions apply when experiment repetitions are omitted");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().repetitions = 3;
        cfg.experiments.front().seed = 9001ULL;
        cfg.experiments.front().algorithm_budget = hpoea::config::BudgetConfig{25, std::nullopt};
        cfg.experiments.front().optimizer_budget = hpoea::config::BudgetConfig{std::nullopt, 800};
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, result.ok() && result.runs.size() == 3,
                       "seeded budgeted experiment expands");
        if (!result.ok() || result.runs.size() != 3) {
            return runner.summarize("suite_expander_tests");
        }
        HPOEA_V2_CHECK(runner, result.runs[0].seed == 9001ULL, "explicit seed is used for repetition zero");
        HPOEA_V2_CHECK(runner, result.runs[1].seed == 9002ULL, "explicit seed increments by repetition");
        HPOEA_V2_CHECK(runner, result.runs[0].algorithm_budget.generations == std::optional<std::size_t>{25},
                       "algorithm budget is copied to runs");
        HPOEA_V2_CHECK(runner, result.runs[0].optimizer_budget.function_evaluations
                                   == std::optional<std::size_t>{800},
                       "optimizer budget is copied to runs");
        HPOEA_V2_CHECK(runner, result.runs[0].planned_output_path
                                   == std::filesystem::path{"results/valid_suite/experiments/sphere_ea/run-000.jsonl"},
                       "planned output path is deterministic");
    }

    {
        auto cfg = make_valid_suite();
        cfg.repetitions = 3;
        const auto first = hpoea::config::expand_suite_config(cfg);
        const auto second = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, first.ok() && second.ok(), "repeated expansion succeeds");
        if (!first.ok() || !second.ok()) {
            return runner.summarize("suite_expander_tests");
        }
        HPOEA_V2_CHECK(runner, first.runs.size() == second.runs.size(),
                       "repeated expansion returns equal run counts");
        if (first.runs.size() != second.runs.size()) {
            return runner.summarize("suite_expander_tests");
        }
        bool same = true;
        for (std::size_t i = 0; i < first.runs.size(); ++i) {
            if (first.runs[i].run_id != second.runs[i].run_id
                || first.runs[i].seed != second.runs[i].seed
                || first.runs[i].planned_output_path != second.runs[i].planned_output_path) {
                same = false;
                break;
            }
        }
        HPOEA_V2_CHECK(runner, same, "run ids, seeds, and paths are deterministic");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().output_name = "shared_output";
        auto second = cfg.experiments.front();
        second.id = "sphere_ea_2";
        second.output_name = "shared_output";
        cfg.experiments.push_back(second);
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[1].output_name"),
                       "duplicate output names fail expansion");
        HPOEA_V2_CHECK(runner, result.runs.empty(), "failed expansion returns no runs");
    }

    {
        auto cfg = make_valid_suite();
        cfg.repetitions = 1;
        cfg.experiments.front().id = "alpha beta";
        cfg.experiments.front().output_name = "alpha_one";
        auto second = cfg.experiments.front();
        second.id = "alpha_beta";
        second.output_name = "alpha_two";
        cfg.experiments.push_back(second);
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "resolved_runs[1].run_id"),
                       "duplicate generated run ids fail expansion");
        HPOEA_V2_CHECK(runner, result.runs.empty(), "run-id collisions return no runs");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().repetitions = 3;
        cfg.experiments.front().seed = std::numeric_limits<std::uint64_t>::max() - 1;
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].seed"),
                       "explicit seed overflow fails expansion");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().output_name = "../escape";
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].output_name"),
                       "unsafe output names fail expansion");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().id = "bad name";
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].id"),
                       "unsafe implicit output names fail expansion");
    }

    {
        auto cfg = make_valid_suite();
        const auto unique_dir = std::filesystem::temp_directory_path() / "hpoea_suite_expander_no_create";
        std::filesystem::remove_all(unique_dir);
        cfg.output_dir = unique_dir;
        const auto result = hpoea::config::expand_suite_config(cfg);
        HPOEA_V2_CHECK(runner, result.ok(), "expansion succeeds for a missing output directory");
        HPOEA_V2_CHECK(runner, !std::filesystem::exists(unique_dir),
                       "expansion does not create directories or files");
    }

    {
        const auto parsed = hpoea::config::parse_config_file(example_path("basic_experiment.toml"));
        HPOEA_V2_CHECK(runner, parsed.ok() && parsed.config.has_value(),
                       "basic example parses before expansion");
        if (!parsed.ok() || !parsed.config.has_value()) {
            return runner.summarize("suite_expander_tests");
        }
        const auto result = hpoea::config::expand_suite_config(*parsed.config);
        HPOEA_V2_CHECK(runner, result.ok(), "basic example expands");
        HPOEA_V2_CHECK(runner, result.runs.size() == 3,
                       "basic example expands one experiment over three repetitions");
    }

    return runner.summarize("suite_expander_tests");
}
