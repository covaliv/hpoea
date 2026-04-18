#include "test_harness.hpp"

#include "hpoea/config/config_parser.hpp"
#include "hpoea/config/config_validator.hpp"

#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::ExperimentSpec;
using hpoea::config::OptimizerSpec;
using hpoea::config::ProblemSpec;
using hpoea::config::SearchParameterMode;
using hpoea::config::SearchParameterSpec;
using hpoea::config::SuiteConfig;
using hpoea::config::ValidationResult;

#if defined(HPOEA_CONFIG_HAS_PAGMO)
std::filesystem::path example_path(const std::string &file_name) {
    return std::filesystem::path{HPOEA_PROJECT_SOURCE_DIR} / "examples" / "configs" / file_name;
}
#endif

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

std::optional<SuiteConfig> parse_config(std::string_view text) {
    const auto result = hpoea::config::parse_config_string(text, "validator.toml");
    if (!result.ok() || !result.config.has_value()) {
        return std::nullopt;
    }
    return *result.config;
}

} // namespace

int main() {
    hpoea::tests_v2::TestRunner runner;

    {
        const auto result = hpoea::config::validate_suite_config(make_valid_suite());
        HPOEA_V2_CHECK(runner, result.ok(), "valid basic config validates");
    }

    {
        auto cfg = make_valid_suite();
        cfg.schema_version = 2;
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "schema_version"),
                       "unsupported schema versions fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.name.clear();
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "suite.name"), "empty suite names fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.output_dir.clear();
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "suite.output_dir"), "empty output_dir fails");
    }

    {
        auto cfg = make_valid_suite();
        cfg.repetitions = 0;
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "suite.repetitions"),
                       "invalid suite repetitions fail");
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
                       "invalid budget values fail");
    }

    {
        struct Case {
            const char *path;
            const char *message;
            void (*mutate)(SuiteConfig &);
        };
        const std::vector<Case> cases{
            {"problems.sphere10", "duplicate problem ids fail", [](SuiteConfig &cfg) {
                 cfg.problems.push_back(cfg.problems.front());
             }},
            {"algorithms.ea_default", "duplicate algorithm ids fail", [](SuiteConfig &cfg) {
                 cfg.algorithms.push_back(cfg.algorithms.front());
             }},
            {"optimizers.optimizer_fast", "duplicate optimizer ids fail", [](SuiteConfig &cfg) {
                 cfg.optimizers.push_back(cfg.optimizers.front());
             }},
            {"experiments[1].id", "duplicate experiment ids fail", [](SuiteConfig &cfg) {
                 cfg.experiments.push_back(cfg.experiments.front());
             }},
            {"experiments[0].problem", "unknown problem refs fail", [](SuiteConfig &cfg) {
                 cfg.experiments.front().problem = "missing_problem";
             }},
            {"experiments[0].algorithm", "unknown algorithm refs fail", [](SuiteConfig &cfg) {
                 cfg.experiments.front().algorithm = "missing_algorithm";
             }},
            {"experiments[0].optimizer", "unknown optimizer refs fail", [](SuiteConfig &cfg) {
                 cfg.experiments.front().optimizer = "missing_optimizer";
             }}
        };

        for (const auto &test_case : cases) {
            auto cfg = make_valid_suite();
            test_case.mutate(cfg);
            const auto result = hpoea::config::validate_suite_config(cfg);
            HPOEA_V2_CHECK(runner, has_error_path(result, test_case.path), test_case.message);
        }
    }

    {
        auto cfg = make_valid_suite();
        SearchParameterSpec spec;
        spec.mode = SearchParameterMode::Range;
        spec.continuous_range = hpoea::core::ContinuousRange{0.9, 0.4};
        cfg.algorithms.front().search_parameters.emplace("scaling_factor", spec);
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.search.scaling_factor"),
                       "range search min greater than max fails");
    }

    {
        auto cfg = make_valid_suite();
        SearchParameterSpec spec;
        spec.mode = SearchParameterMode::Range;
        cfg.algorithms.front().search_parameters.emplace("scaling_factor", spec);
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.search.scaling_factor"),
                       "range search without bounds fails");
    }

    {
        const auto cfg = parse_config(R"(
            schema_version = 1

            [suite]
            name = "partial_bounds"
            output_dir = "results/partial_bounds"

            [problems.sphere10]
            type = "sphere"
            dimension = 10

            [algorithms.ea_default]
            type = "research_ea"

            [algorithms.ea_default.search.scaling_factor]
            mode = "range"
            min = 0.4

            [optimizers.optimizer_fast]
            type = "research_optimizer"

            [[experiments]]
            id = "sphere_ea"
            problem = "sphere10"
            algorithm = "ea_default"
            optimizer = "optimizer_fast"
        )");
        HPOEA_V2_CHECK(runner, cfg.has_value(), "partial search bounds parse for validation");
        if (!cfg.has_value()) {
            return runner.summarize("config_validator_tests");
        }
        const auto result = hpoea::config::validate_suite_config(*cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.search.scaling_factor"),
                       "partial range bounds fail");
    }

    {
        auto cfg = make_valid_suite();
        SearchParameterSpec spec;
        spec.mode = SearchParameterMode::IntegerRange;
        spec.integer_range = hpoea::core::IntegerRange{10, 5};
        cfg.algorithms.front().search_parameters.emplace("population_size", spec);
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.search.population_size"),
                       "integer_range min greater than max fails");
    }

    {
        auto cfg = make_valid_suite();
        SearchParameterSpec spec;
        spec.mode = SearchParameterMode::Choice;
        cfg.algorithms.front().search_parameters.emplace("variant", spec);
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.search.variant.values"),
                       "empty choice search values fail");
    }

    {
        auto cfg = make_valid_suite();
        SearchParameterSpec spec;
        spec.mode = SearchParameterMode::Exclude;
        spec.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
        cfg.algorithms.front().search_parameters.emplace("ftol", spec);
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.search.ftol"),
                       "exclude search bounds fail");
    }


    {
        auto cfg = make_valid_suite();
        SearchParameterSpec spec;
        spec.mode = SearchParameterMode::Range;
        spec.continuous_range = hpoea::core::ContinuousRange{0.1, 0.9};
        spec.choices.push_back(0.5);
        cfg.algorithms.front().search_parameters.emplace("scaling_factor", spec);
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.search.scaling_factor.values"),
                       "values on range search fail validation");
    }

    {
        auto cfg = make_valid_suite();
        cfg.algorithms.front().fixed_parameters.emplace("scaling_factor", 0.5);
        SearchParameterSpec spec;
        spec.mode = SearchParameterMode::Range;
        spec.continuous_range = hpoea::core::ContinuousRange{0.1, 0.9};
        cfg.algorithms.front().search_parameters.emplace("scaling_factor", spec);
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.search.scaling_factor"),
                       "fixed and search parameter conflicts fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().output_name = "../escape";
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].output_name"),
                       "unsafe output names fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().id = "bad name";
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].id"),
                       "unsafe implicit output names fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().output_name = "shared_output";
        auto second = cfg.experiments.front();
        second.id = "sphere_ea_2";
        second.output_name = "shared_output";
        cfg.experiments.push_back(second);
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[1].output_name"),
                       "duplicate output names fail");
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
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "resolved_runs[1].run_id"),
                       "duplicate generated run ids fail");
    }

    {
        auto cfg = make_valid_suite();
        cfg.experiments.front().repetitions = 3;
        cfg.experiments.front().seed = std::numeric_limits<std::uint64_t>::max() - 1;
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "experiments[0].seed"),
                       "explicit seed overflow fails");
    }

#if !defined(HPOEA_CONFIG_HAS_PAGMO)
    {
        auto cfg = make_valid_suite();
        cfg.algorithms.front().type = "de";
        cfg.optimizers.front().type = "cmaes";
        const auto result = hpoea::config::validate_suite_config(cfg);
        HPOEA_V2_CHECK(runner, has_error_path(result, "algorithms.ea_default.type"),
                       "core-only builds reject known Pagmo algorithm types");
        HPOEA_V2_CHECK(runner, has_error_path(result, "optimizers.optimizer_fast.type"),
                       "core-only builds reject known Pagmo optimizer types");
    }
#endif

#if defined(HPOEA_CONFIG_HAS_PAGMO)
    {
        const auto parsed = hpoea::config::parse_config_file(example_path("basic_experiment.toml"));
        HPOEA_V2_CHECK(runner, parsed.ok() && parsed.config.has_value(),
                       "basic example parses in Pagmo-enabled validation lane");
        if (!parsed.ok() || !parsed.config.has_value()) {
            return runner.summarize("config_validator_tests");
        }
        const auto result = hpoea::config::validate_suite_config(*parsed.config);
        HPOEA_V2_CHECK(runner, result.ok(), "Pagmo-enabled builds accept known Pagmo types");
    }
#endif

    return runner.summarize("config_validator_tests");
}
