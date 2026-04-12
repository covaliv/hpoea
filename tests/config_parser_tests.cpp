#include "test_harness.hpp"

#include "hpoea/config/config_parser.hpp"
#include "hpoea/config/config_validator.hpp"

int main() {
    hpoea::tests_v2::TestRunner runner;

    const auto result = hpoea::config::parse_config_string(R"(
        schema_version = 1

        [suite]
        name = "basic_suite"
        output_dir = "results/basic_suite"
        repetitions = 2

        [problems.sphere10]
        type = "sphere"

        [algorithms.ea_default]
        type = "research_ea"

        [optimizers.optimizer_fast]
        type = "research_optimizer"

        [[experiments]]
        id = "sphere_ea"
        problem = "sphere10"
        algorithm = "ea_default"
        optimizer = "optimizer_fast"
    )");

    HPOEA_V2_CHECK(runner, result.ok(), "minimal config parses");
    HPOEA_V2_CHECK(runner, result.config.has_value(), "minimal config returns suite");
    if (!result.config.has_value()) {
        return runner.summarize("config_parser_tests");
    }
    HPOEA_V2_CHECK(runner, result.config->name == "basic_suite", "suite name is parsed");
    HPOEA_V2_CHECK(runner, result.config->experiments.size() == 1, "explicit experiment is parsed");

    const auto validation = hpoea::config::validate_suite_config(*result.config);
    HPOEA_V2_CHECK(runner, validation.ok(), "minimal config validates");

    return runner.summarize("config_parser_tests");
}
