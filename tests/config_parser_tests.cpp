#include "test_harness.hpp"

#include "hpoea/config/config_parser.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace {

std::filesystem::path example_path(const std::string &file_name) {
    return std::filesystem::path{HPOEA_PROJECT_SOURCE_DIR} / "examples" / "configs" / file_name;
}

bool has_error_path(const hpoea::config::ParseResult &result,
                    const std::string &path) {
    for (const auto &diag : result.diagnostics) {
        if (diag.path == path) {
            return true;
        }
    }
    return false;
}

const char *basic_config() {
    return R"(
        schema_version = 1

        [suite]
        name = "basic"
        output_dir = "results/basic"
        suite_seed = 1234
        repetitions = 2

        [problems.sphere10]
        type = "sphere"
        dimension = 10
        lower_bound = -5.0
        upper_bound = 5.0

        [algorithms.de_default]
        type = "de"
        fixed = { population_size = 40, variant = 2, label = "draft" }

        [algorithms.de_default.search.scaling_factor]
        mode = "range"
        min = 0.4
        max = 0.9

        [algorithms.de_default.search.variant]
        mode = "integer_range"
        min = 1
        max = 4

        [algorithms.de_default.search.crossover_rate]
        mode = "choice"
        values = [0.5, 0.7, 0.9]

        [algorithms.de_default.search.ftol]
        mode = "exclude"

        [optimizers.cmaes_fast]
        type = "cmaes"
        parameters = { generations = 10, sigma0 = 0.3 }

        [[experiments]]
        id = "sphere_de"
        problem = "sphere10"
        algorithm = "de_default"
        optimizer = "cmaes_fast"
        repetitions = 3
        seed = 9001
        output_name = "sphere_de"

        [experiments.algorithm_budget]
        generations = 25

        [experiments.optimizer_budget]
        function_evaluations = 800
    )";
}

} // namespace

int main() {
    hpoea::tests_v2::TestRunner runner;

    {
        const auto result = hpoea::config::parse_config_string(basic_config(), "basic.toml");
        HPOEA_V2_CHECK(runner, result.ok(), "valid basic config parses");
        HPOEA_V2_CHECK(runner, result.config.has_value(), "valid basic config returns a suite config");
        if (!result.config.has_value()) {
            return runner.summarize("config_parser_tests");
        }

        const auto &cfg = *result.config;
        HPOEA_V2_CHECK(runner, cfg.name == "basic", "suite name parses");
        HPOEA_V2_CHECK(runner, cfg.output_dir == std::filesystem::path{"results/basic"},
                       "suite output_dir parses");
        HPOEA_V2_CHECK(runner, cfg.suite_seed == std::optional<std::uint64_t>{1234},
                       "suite_seed parses");
        HPOEA_V2_CHECK(runner, cfg.repetitions == 2, "suite repetitions parse");
        HPOEA_V2_CHECK(runner, cfg.problems.size() == 1, "problem definition parses");
        HPOEA_V2_CHECK(runner, cfg.algorithms.size() == 1, "algorithm definition parses");
        HPOEA_V2_CHECK(runner, cfg.optimizers.size() == 1, "optimizer definition parses");
        HPOEA_V2_CHECK(runner, cfg.experiments.size() == 1, "explicit experiment parses");
    }

    {
        const auto result = hpoea::config::parse_config_file(example_path("basic_experiment.toml"));
        HPOEA_V2_CHECK(runner, result.ok(), "basic example config parses");
    }

    {
        const auto result = hpoea::config::parse_config_string(basic_config(), "values.toml");
        HPOEA_V2_CHECK(runner, result.ok() && result.config.has_value(), "basic config is available for value checks");
        if (!result.ok() || !result.config.has_value()) {
            return runner.summarize("config_parser_tests");
        }
        const auto &problem_params = result.config->problems.front().parameters;
        const auto &fixed = result.config->algorithms.front().fixed_parameters;
        const auto &search = result.config->algorithms.front().search_parameters;
        const auto &optimizer_params = result.config->optimizers.front().parameters;
        const auto &experiment = result.config->experiments.front();

        HPOEA_V2_CHECK(runner, std::holds_alternative<std::int64_t>(problem_params.at("dimension")),
                       "problem integer parameter parses");
        HPOEA_V2_CHECK(runner, std::holds_alternative<double>(problem_params.at("lower_bound")),
                       "problem floating-point parameter parses");
        HPOEA_V2_CHECK(runner, std::holds_alternative<std::int64_t>(fixed.at("population_size")),
                       "algorithm fixed integer parses");
        HPOEA_V2_CHECK(runner, std::holds_alternative<std::string>(fixed.at("label")),
                       "algorithm fixed string parses");
        HPOEA_V2_CHECK(runner, search.at("scaling_factor").mode == hpoea::config::SearchParameterMode::Range,
                       "range search mode parses");
        HPOEA_V2_CHECK(runner, search.at("variant").mode == hpoea::config::SearchParameterMode::IntegerRange,
                       "integer_range search mode parses");
        HPOEA_V2_CHECK(runner, search.at("crossover_rate").choices.size() == 3,
                       "choice search values parse");
        HPOEA_V2_CHECK(runner, search.at("ftol").mode == hpoea::config::SearchParameterMode::Exclude,
                       "exclude search mode parses");
        HPOEA_V2_CHECK(runner, std::holds_alternative<double>(optimizer_params.at("sigma0")),
                       "optimizer parameter parses");
        HPOEA_V2_CHECK(runner, experiment.algorithm_budget.has_value(),
                       "algorithm budget parses");
        HPOEA_V2_CHECK(runner, experiment.optimizer_budget.has_value(),
                       "optimizer budget parses");
    }

    {
        const auto result = hpoea::config::parse_config_string(R"(
            schema_version = 1

            [suite]
            name = "arrays"
            output_dir = "results/arrays"

            [problems.knapsack_small]
            type = "knapsack"
            values = [10, 7.5, 3]
            weights = [5, 3, 1]
            capacity = 6
        )", "arrays.toml");
        HPOEA_V2_CHECK(runner, result.ok() && result.config.has_value(), "numeric problem arrays parse");
        if (!result.ok() || !result.config.has_value()) {
            return runner.summarize("config_parser_tests");
        }
        const auto &params = result.config->problems.front().parameters;
        HPOEA_V2_CHECK(runner, std::holds_alternative<std::vector<double>>(params.at("values")),
                       "mixed numeric arrays are stored as floating-point arrays");
        HPOEA_V2_CHECK(runner, std::holds_alternative<std::vector<std::int64_t>>(params.at("weights")),
                       "integer arrays are preserved");
    }

    {
        const auto result = hpoea::config::parse_config_string("[suite\nname = \"broken\"", "broken.toml");
        HPOEA_V2_CHECK(runner, !result.ok(), "malformed TOML fails");
        HPOEA_V2_CHECK(runner, !result.diagnostics.empty(), "malformed TOML reports diagnostics");
    }


    {
        const auto result = hpoea::config::parse_config_string(R"(
            schema_version = 1

            [suite]
            name = "range_values"
            output_dir = "results/range_values"

            [algorithms.ea_default]
            type = "research_ea"

            [algorithms.ea_default.search.scaling_factor]
            mode = "range"
            min = 0.1
            max = 0.9
            values = [0.5]
        )", "range_values.toml");
        HPOEA_V2_CHECK(runner, result.ok() && result.config.has_value(),
                       "search shape errors are left to validation");
    }

    {
        const auto result = hpoea::config::parse_config_string(R"(
            schema_version = 1

            [suite]
            name = "removed_fields"
            output_dir = "results/removed_fields"
            resume = true
            max_parallel_experiments = 4
        )", "removed_fields.toml");
        HPOEA_V2_CHECK(runner, !result.ok(), "removed suite fields fail");
        HPOEA_V2_CHECK(runner, has_error_path(result, "suite.resume"), "resume path is reported");
        HPOEA_V2_CHECK(runner, has_error_path(result, "suite.max_parallel_experiments"),
                       "max_parallel_experiments path is reported");
    }

    {
        const auto result = hpoea::config::parse_config_string(R"(
            schema_version = 1

            [suite]
            name = "defaults"
            output_dir = "results/defaults"

            [suite.defaults.algorithm_budget]
            generations = 50
        )", "defaults.toml");
        HPOEA_V2_CHECK(runner, !result.ok(), "removed suite defaults fail");
        HPOEA_V2_CHECK(runner, has_error_path(result, "suite.defaults"), "suite defaults path is reported");
    }

    {
        const auto result = hpoea::config::parse_config_string(R"(
            schema_version = 1

            [suite]
            name = "matrices"
            output_dir = "results/matrices"

            [[matrices]]
            id = "matrix"
            problems = ["sphere10"]
        )", "matrices.toml");
        HPOEA_V2_CHECK(runner, !result.ok(), "matrix experiments fail");
        HPOEA_V2_CHECK(runner, has_error_path(result, "matrices"), "matrices path is reported");
    }

    {
        const auto result = hpoea::config::parse_config_string(R"(
            schema_version = 1

            [suite]
            name = "wrong_type"
            output_dir = "results/wrong_type"
            repetitions = "three"
        )", "wrong_type.toml");
        HPOEA_V2_CHECK(runner, !result.ok(), "wrong suite value types fail");
        HPOEA_V2_CHECK(runner, has_error_path(result, "suite.repetitions"),
                       "wrong suite value path is reported");
    }

    {
        const auto result = hpoea::config::parse_config_string(R"(
            schema_version = 1

            [suite]
            name = "bad_arrays"
            output_dir = "results/bad_arrays"

            [problems.bad]
            type = "sphere"
            values = [1, "two"]
            bounds = { min = -5.0, max = 5.0 }
        )", "bad_arrays.toml");
        HPOEA_V2_CHECK(runner, !result.ok(), "mixed arrays and nested parameter tables fail");
        HPOEA_V2_CHECK(runner, has_error_path(result, "problems.bad.values[1]"),
                       "mixed array element path is reported");
        HPOEA_V2_CHECK(runner, has_error_path(result, "problems.bad.bounds"),
                       "nested table path is reported");
    }

    return runner.summarize("config_parser_tests");
}
