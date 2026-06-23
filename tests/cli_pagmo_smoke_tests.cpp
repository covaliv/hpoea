#include "test_harness.hpp"
#include "cli_util.hpp"

#include <algorithm>
#include <filesystem>
#include <string>

using namespace hpoea::tests_v2;

namespace {

std::string tiny_pagmo_config(const std::filesystem::path &output_dir) {
    return R"(schema_version = 1

[suite]
name = "tiny_cli_run"
output_dir = ")" + output_dir.generic_string() + R"("
suite_seed = 123
repetitions = 1

[problems.sphere2]
type = "sphere"
dimension = 2
lower_bound = -1.0
upper_bound = 1.0

[algorithms.de_tiny]
type = "de"
fixed = { population_size = 5, variant = 2, generations = 1, ftol = 1e-6, xtol = 1e-6 }

[algorithms.de_tiny.search.scaling_factor]
mode = "range"
min = 0.5
max = 0.9

[algorithms.de_tiny.search.crossover_rate]
mode = "range"
min = 0.8
max = 0.9

[optimizers.cmaes_tiny]
type = "cmaes"
parameters = { generations = 1, sigma0 = 0.3, ftol = 1e-6, xtol = 1e-6 }

[[experiments]]
id = "sphere_de_cmaes"
problem = "sphere2"
algorithm = "de_tiny"
optimizer = "cmaes_tiny"
repetitions = 1
seed = 7
output_name = "tiny"

[experiments.algorithm_budget]
generations = 1
function_evaluations = 10

[experiments.optimizer_budget]
generations = 1
function_evaluations = 16
)";
}

std::string tiny_random_search_config(const std::filesystem::path &output_dir) {
    return R"(schema_version = 1

[suite]
name = "tiny_random_search_cli_run"
output_dir = ")" + output_dir.generic_string() + R"("
suite_seed = 123
repetitions = 1

[problems.sphere2]
type = "sphere"
dimension = 2
lower_bound = -1.0
upper_bound = 1.0

[algorithms.de_tiny]
type = "de"
fixed = { population_size = 5, variant = 2, crossover_rate = 0.9, generations = 1, ftol = 1e-6, xtol = 1e-6 }

[algorithms.de_tiny.search.scaling_factor]
mode = "range"
min = 0.5
max = 0.9

[optimizers.random_tiny]
type = "random_search"
parameters = { sample_count = 2 }

[[experiments]]
id = "sphere_de_random"
problem = "sphere2"
algorithm = "de_tiny"
optimizer = "random_tiny"
repetitions = 1
seed = 17
output_name = "tiny-random"

[experiments.algorithm_budget]
generations = 1
function_evaluations = 10

[experiments.optimizer_budget]
function_evaluations = 2
)";
}

} // namespace

int main() {
    hpoea::tests_v2::TestRunner runner;
    const auto work_dir = unique_test_dir("cli_pagmo_smoke_tests");
    std::filesystem::remove_all(work_dir);
    std::filesystem::create_directories(work_dir);

    const auto output_dir = work_dir / "out";
    const auto config_path = work_dir / "tiny_run.toml";
    write_file(config_path, tiny_pagmo_config(output_dir));

    const auto result = run_cli({"run", config_path.string()}, work_dir);
    HPOEA_V2_CHECK(runner, result.exit_code == 0, "Pagmo CLI run exits successfully");
    HPOEA_V2_CHECK(runner, contains(result.stdout_text, "ran: sphere_de_cmaes__rep000"),
                   "Pagmo CLI run reports completed run");
    HPOEA_V2_CHECK(runner, result.stderr_text.empty(), "Pagmo CLI run has no stderr");

    const auto log_path = output_dir / "experiments" / "tiny" / "run-000.jsonl";
    HPOEA_V2_CHECK(runner, std::filesystem::exists(log_path), "Pagmo CLI run creates JSONL log");
    const auto log_text = read_file(log_path);
    HPOEA_V2_CHECK(runner, contains(log_text, "\"schema_version\":3"),
                   "Pagmo CLI run writes schema version");
    HPOEA_V2_CHECK(runner, contains(log_text, "\"problem_id\":\"sphere\""),
                   "Pagmo CLI run logs sphere problem");
    HPOEA_V2_CHECK(runner, contains(log_text, "\"implementation\":\"pagmo::cmaes\""),
                   "Pagmo CLI run logs CMA-ES optimizer");
    HPOEA_V2_CHECK(runner, contains(log_text, "\"population_size\":5"),
                   "Pagmo CLI run logs fixed DE parameter");

    const auto rerun_result = run_cli({"run", config_path.string()}, work_dir);
    HPOEA_V2_CHECK(runner, rerun_result.exit_code == 0, "Pagmo CLI rerun exits successfully");
    const auto rerun_text = read_file(log_path);
    HPOEA_V2_CHECK(runner,
                   std::count(rerun_text.begin(), rerun_text.end(), '\n') ==
                       std::count(log_text.begin(), log_text.end(), '\n'),
                   "rerunning a config replaces the JSONL records instead of appending");

    const auto random_output_dir = work_dir / "random-out";
    const auto random_config_path = work_dir / "tiny_random_run.toml";
    write_file(random_config_path, tiny_random_search_config(random_output_dir));

    const auto random_result = run_cli({"run", random_config_path.string()}, work_dir);
    HPOEA_V2_CHECK(runner, random_result.exit_code == 0,
                   "Random Search CLI run exits successfully");
    HPOEA_V2_CHECK(runner, contains(random_result.stdout_text, "ran: sphere_de_random__rep000"),
                   "Random Search CLI run reports completed run");
    HPOEA_V2_CHECK(runner, contains(random_result.stdout_text, "records: 2"),
                   "Random Search CLI run reports one record per sample");
    HPOEA_V2_CHECK(runner, random_result.stderr_text.empty(),
                   "Random Search CLI run has no stderr");

    const auto random_log_path = random_output_dir / "experiments" / "tiny-random" / "run-000.jsonl";
    HPOEA_V2_CHECK(runner, std::filesystem::exists(random_log_path),
                   "Random Search CLI run creates JSONL log");
    const auto random_log_text = read_file(random_log_path);
    HPOEA_V2_CHECK(runner, contains(random_log_text, "\"family\":\"RandomSearch\""),
                   "Random Search CLI run logs optimizer family");
    HPOEA_V2_CHECK(runner, contains(random_log_text, "\"implementation\":\"uniform_random\""),
                   "Random Search CLI run logs optimizer implementation");

    std::filesystem::remove_all(work_dir);
    return runner.summarize("cli_pagmo_smoke_tests");
}
