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

bool has_error(const hpoea::config::ParseResult &result,
               const std::string &source,
               const std::string &path,
               const std::string &message) {
    for (const auto &diag : result.diagnostics) {
        if (diag.source == source && diag.path == path
            && diag.message.find(message) != std::string::npos
            && diag.severity == hpoea::config::ParseDiagnosticSeverity::Error) {
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
        HPOEA_V2_CHECK(runner, cfg.problems.front().id == "sphere10", "problem id parses");
        HPOEA_V2_CHECK(runner, cfg.problems.front().type == "sphere", "problem type parses");
        HPOEA_V2_CHECK(runner, cfg.algorithms.front().id == "de_default", "algorithm id parses");
        HPOEA_V2_CHECK(runner, cfg.algorithms.front().type == "de", "algorithm type parses");
        HPOEA_V2_CHECK(runner, cfg.optimizers.front().id == "cmaes_fast", "optimizer id parses");
        HPOEA_V2_CHECK(runner, cfg.optimizers.front().type == "cmaes", "optimizer type parses");
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

        HPOEA_V2_CHECK(runner, std::get<std::int64_t>(problem_params.at("dimension")) == 10,
                       "problem integer parameter value parses");
        HPOEA_V2_CHECK(runner, std::get<double>(problem_params.at("lower_bound")) == -5.0,
                       "problem lower_bound value parses");
        HPOEA_V2_CHECK(runner, std::get<double>(problem_params.at("upper_bound")) == 5.0,
                       "problem upper_bound value parses");
        HPOEA_V2_CHECK(runner, std::get<std::int64_t>(fixed.at("population_size")) == 40,
                       "algorithm fixed integer value parses");
        HPOEA_V2_CHECK(runner, std::get<std::int64_t>(fixed.at("variant")) == 2,
                       "algorithm fixed variant value parses");
        HPOEA_V2_CHECK(runner, std::get<std::string>(fixed.at("label")) == "draft",
                       "algorithm fixed string value parses");
        HPOEA_V2_CHECK(runner, search.at("scaling_factor").mode == hpoea::config::SearchParameterMode::Range,
                       "range search mode parses");
        HPOEA_V2_CHECK(runner, search.at("scaling_factor").continuous_range->lower == 0.4
                                  && search.at("scaling_factor").continuous_range->upper == 0.9,
                       "range search bounds parse");
        HPOEA_V2_CHECK(runner, search.at("variant").mode == hpoea::config::SearchParameterMode::IntegerRange,
                       "integer_range search mode parses");
        HPOEA_V2_CHECK(runner, search.at("variant").integer_range->lower == 1
                                  && search.at("variant").integer_range->upper == 4,
                       "integer_range search bounds parse");
        HPOEA_V2_CHECK(runner, search.at("crossover_rate").choices.size() == 3,
                       "choice search values parse");
        HPOEA_V2_CHECK(runner, std::get<double>(search.at("crossover_rate").choices[0]) == 0.5
                                  && std::get<double>(search.at("crossover_rate").choices[1]) == 0.7
                                  && std::get<double>(search.at("crossover_rate").choices[2]) == 0.9,
                       "choice search values preserve order");
        HPOEA_V2_CHECK(runner, search.at("ftol").mode == hpoea::config::SearchParameterMode::Exclude,
                       "exclude search mode parses");
        HPOEA_V2_CHECK(runner, std::get<std::int64_t>(optimizer_params.at("generations")) == 10,
                       "optimizer generations value parses");
        HPOEA_V2_CHECK(runner, std::get<double>(optimizer_params.at("sigma0")) == 0.3,
                       "optimizer floating parameter value parses");
        HPOEA_V2_CHECK(runner, experiment.id == "sphere_de" && experiment.problem == "sphere10"
                                  && experiment.algorithm == "de_default"
                                  && experiment.optimizer == "cmaes_fast",
                       "experiment references parse exactly");
        HPOEA_V2_CHECK(runner, experiment.repetitions == std::optional<std::size_t>{3},
                       "experiment repetitions parse");
        HPOEA_V2_CHECK(runner, experiment.seed == std::optional<std::uint64_t>{9001},
                       "experiment seed parses");
        HPOEA_V2_CHECK(runner, experiment.output_name == std::optional<std::string>{"sphere_de"},
                       "experiment output_name parses");
        HPOEA_V2_CHECK(runner, experiment.algorithm_budget->generations == std::optional<std::size_t>{25},
                       "algorithm budget generation value parses");
        HPOEA_V2_CHECK(runner, experiment.optimizer_budget->function_evaluations == std::optional<std::size_t>{800},
                       "optimizer budget function evaluation value parses");
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
        HPOEA_V2_CHECK(runner, std::get<std::vector<double>>(params.at("values"))
                                   == std::vector<double>({10.0, 7.5, 3.0}),
                       "mixed numeric arrays are stored as floating-point arrays");
        HPOEA_V2_CHECK(runner, std::get<std::vector<std::int64_t>>(params.at("weights"))
                                   == std::vector<std::int64_t>({5, 3, 1}),
                       "integer arrays preserve exact values");
        HPOEA_V2_CHECK(runner, std::get<std::int64_t>(params.at("capacity")) == 6,
                       "scalar parameter adjacent to arrays parses");
    }

    {
        const auto result = hpoea::config::parse_config_string("[suite\nname = \"broken\"", "broken.toml");
        HPOEA_V2_CHECK(runner, !result.ok(), "malformed TOML fails");
        HPOEA_V2_CHECK(runner, result.diagnostics.size() == 1, "malformed TOML reports one diagnostic");
        HPOEA_V2_CHECK(runner, result.diagnostics.front().source == "broken.toml",
                       "malformed TOML diagnostic preserves source");
        HPOEA_V2_CHECK(runner, result.diagnostics.front().severity == hpoea::config::ParseDiagnosticSeverity::Error,
                       "malformed TOML diagnostic is an error");
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
        HPOEA_V2_CHECK(runner, has_error(result, "removed_fields.toml", "suite.resume", "unknown field"),
                       "resume diagnostic is exact");
        HPOEA_V2_CHECK(runner, has_error(result, "removed_fields.toml", "suite.max_parallel_experiments",
                                         "unknown field"),
                       "max_parallel_experiments diagnostic is exact");
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
        HPOEA_V2_CHECK(runner, has_error(result, "defaults.toml", "suite.defaults", "unknown field"),
                       "suite defaults diagnostic is exact");
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
        HPOEA_V2_CHECK(runner, has_error(result, "matrices.toml", "matrices",
                                         "unsupported section 'matrices'; see examples/configs/basic_experiment.toml"),
                       "matrices diagnostic is exact");
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
        HPOEA_V2_CHECK(runner, has_error(result, "wrong_type.toml", "suite.repetitions",
                                         "expected integer, got string"),
                       "wrong suite value diagnostic is exact");
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
        HPOEA_V2_CHECK(runner, has_error(result, "bad_arrays.toml", "problems.bad.values[1]",
                                         "mixed-type arrays are not supported"),
                       "mixed array element diagnostic is exact");
        HPOEA_V2_CHECK(runner, has_error(result, "bad_arrays.toml", "problems.bad.bounds",
                                         "unsupported value type: table"),
                       "nested table diagnostic is exact");
    }

    return runner.summarize("config_parser_tests");
}
