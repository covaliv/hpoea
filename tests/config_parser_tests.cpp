#include "test_harness.hpp"

#include "hpoea/config/config_parser.hpp"

#include <cstdint>
#include <variant>

int main() {
    hpoea::tests_v2::TestRunner runner;

    const auto result = hpoea::config::parse_config_string(R"(
        schema_version = 1

        [suite]
        name = "explicit_suite"
        output_dir = "results/explicit_suite"
        suite_seed = 1234
        repetitions = 2

        [problems.sphere10]
        type = "sphere"
        dimension = 10
        lower_bound = -5.0
        upper_bound = 5.0

        [algorithms.ea_default]
        type = "research_ea"
        fixed = { population_size = 40, variant = "best" }

        [algorithms.ea_default.search.scaling_factor]
        mode = "range"
        min = 0.4
        max = 0.9

        [optimizers.optimizer_fast]
        type = "research_optimizer"
        parameters = { generations = 10 }

        [[experiments]]
        id = "sphere_ea"
        problem = "sphere10"
        algorithm = "ea_default"
        optimizer = "optimizer_fast"
        repetitions = 3
        seed = 9001
        output_name = "sphere_ea"

        [experiments.algorithm_budget]
        generations = 25
    )");

    HPOEA_V2_CHECK(runner, result.ok(), "explicit config parses");
    HPOEA_V2_CHECK(runner, result.config.has_value(), "explicit config returns suite");
    if (!result.config.has_value()) {
        return runner.summarize("config_parser_tests");
    }
    const auto &cfg = *result.config;
    HPOEA_V2_CHECK(runner, cfg.problems.size() == 1, "problem definition is parsed");
    HPOEA_V2_CHECK(runner, cfg.algorithms.front().fixed_parameters.contains("population_size"),
                   "fixed algorithm parameter is parsed");
    HPOEA_V2_CHECK(runner, cfg.algorithms.front().search_parameters.contains("scaling_factor"),
                   "search parameter is parsed");
    HPOEA_V2_CHECK(runner, cfg.experiments.front().algorithm_budget.has_value(),
                   "experiment budget is parsed");

    return runner.summarize("config_parser_tests");
}
