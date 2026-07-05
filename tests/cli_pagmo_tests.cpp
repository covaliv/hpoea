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
validation_repeats = 1
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

// feval budget 4 below the cmaes initial population of 8
std::string starved_pagmo_config(const std::filesystem::path &output_dir) {
    return R"(schema_version = 1

[suite]
name = "starved_cli_run"
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
parameters = { sigma0 = 0.3 }

[[experiments]]
id = "sphere_de_starved"
problem = "sphere2"
algorithm = "de_tiny"
optimizer = "cmaes_tiny"
repetitions = 1
seed = 7
output_name = "starved"

[experiments.algorithm_budget]
generations = 1
function_evaluations = 10

[experiments.optimizer_budget]
function_evaluations = 4
)";
}

// two runnable matrix cells
std::string matrix_config(const std::filesystem::path &output_dir) {
    return R"(schema_version = 1

[suite]
name = "matrix_cli_run"
output_dir = ")" + output_dir.generic_string() + R"("
suite_seed = 5
repetitions = 1

[problems.rastrigin2]
type = "rastrigin"
dimension = 2

[problems.rosen2]
type = "rosenbrock"
dimension = 2

[algorithms.sade_tiny]
type = "sade"
fixed = { population_size = 8, variant = 2, variant_adptv = 1, ftol = 1e-6, xtol = 1e-6, memory = false }

[algorithms.de_tiny]
type = "de"
fixed = { population_size = 5, variant = 2, generations = 1, ftol = 1e-6, xtol = 1e-6 }

[algorithms.de_tiny.search.crossover_rate]
mode = "range"
min = 0.8
max = 0.9

[algorithms.de_tiny.search.scaling_factor]
mode = "range"
min = 0.5
max = 0.9

[optimizers.sa_tiny]
type = "simulated_annealing"
parameters = { n_T_adj = 1, n_range_adj = 1, bin_size = 1 }

[optimizers.nm_tiny]
type = "nelder_mead"

[[experiments]]
id = "rastrigin_sade_sa"
problem = "rastrigin2"
algorithm = "sade_tiny"
optimizer = "sa_tiny"
seed = 3
output_name = "rastrigin-sade-sa"

[experiments.algorithm_budget]
generations = 1
function_evaluations = 30

[experiments.optimizer_budget]
function_evaluations = 4

[[experiments]]
id = "rosenbrock_de_nm"
problem = "rosen2"
algorithm = "de_tiny"
optimizer = "nm_tiny"
seed = 4
output_name = "rosenbrock-de-nm"

[experiments.algorithm_budget]
generations = 1
function_evaluations = 12

[experiments.optimizer_budget]
function_evaluations = 8
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

std::string baseline_config(const std::filesystem::path &output_dir) {
    return R"(schema_version = 1

[suite]
name = "baseline_cli_run"
output_dir = ")" + output_dir.generic_string() + R"("
suite_seed = 123
repetitions = 1

[problems.sphere2]
type = "sphere"
dimension = 2
lower_bound = -1.0
upper_bound = 1.0

[algorithms.de_base]
type = "de"
fixed = { population_size = 6, generations = 1 }

[optimizers.base]
type = "baseline"

[[experiments]]
id = "sphere_de_baseline"
problem = "sphere2"
algorithm = "de_base"
optimizer = "base"
repetitions = 1
validation_repeats = 2
seed = 9
output_name = "baseline"

[experiments.algorithm_budget]
generations = 1
function_evaluations = 12

[experiments.optimizer_budget]
function_evaluations = 1
)";
}

// every sampled population overspends the inner feval budget
std::string degraded_config(const std::filesystem::path &output_dir) {
    return R"(schema_version = 1

[suite]
name = "degraded_cli_run"
output_dir = ")" + output_dir.generic_string() + R"("
suite_seed = 123
repetitions = 1

[problems.sphere2]
type = "sphere"
dimension = 2
lower_bound = -1.0
upper_bound = 1.0

[algorithms.de_over]
type = "de"
fixed = { variant = 2, generations = 1, crossover_rate = 0.9, scaling_factor = 0.8, ftol = 1e-6, xtol = 1e-6 }

[algorithms.de_over.search.population_size]
mode = "integer_range"
min = 5
max = 8

[optimizers.random_tiny]
type = "random_search"
parameters = { sample_count = 4 }

[[experiments]]
id = "sphere_de_degraded"
problem = "sphere2"
algorithm = "de_over"
optimizer = "random_tiny"
repetitions = 1
seed = 11
output_name = "degraded"

[experiments.algorithm_budget]
generations = 1
function_evaluations = 4

[experiments.optimizer_budget]
function_evaluations = 4
)";
}

} // namespace

int main() {
    hpoea::tests_v2::TestRunner runner;
    const auto work_dir = unique_test_dir("cli_pagmo_tests");
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
    HPOEA_V2_CHECK(runner, contains(log_text, "\"schema_version\":4"),
                   "Pagmo CLI run writes schema version");
    HPOEA_V2_CHECK(runner, contains(log_text, "\"problem_id\":\"sphere\""),
                   "Pagmo CLI run logs sphere problem");
    HPOEA_V2_CHECK(runner, contains(log_text, "\"implementation\":\"pagmo::cmaes\""),
                   "Pagmo CLI run logs CMA-ES optimizer");
    HPOEA_V2_CHECK(runner, contains(log_text, "\"population_size\":5"),
                   "Pagmo CLI run logs fixed DE parameter");
    HPOEA_V2_CHECK(runner, contains(log_text, "\"phase\":\"tuning\""),
                   "Pagmo CLI run logs tuning records");
    HPOEA_V2_CHECK(runner, contains(log_text, "\"phase\":\"validation\""),
                   "Pagmo CLI run logs held-out validation records");

    const auto rerun_result = run_cli({"run", config_path.string()}, work_dir);
    HPOEA_V2_CHECK(runner, rerun_result.exit_code == 0, "Pagmo CLI rerun exits successfully");
    const auto rerun_text = read_file(log_path);
    HPOEA_V2_CHECK(runner,
                   std::count(rerun_text.begin(), rerun_text.end(), '\n') ==
                       std::count(log_text.begin(), log_text.end(), '\n'),
                   "rerunning a config replaces the JSONL records instead of appending");

    const auto stale_run_path = output_dir / "experiments" / "tiny" / "run-042.jsonl";
    write_file(stale_run_path, "{\"stale\":true}\n");
    const auto orphan_dir = output_dir / "experiments" / "orphaned";
    std::filesystem::create_directories(orphan_dir);
    write_file(orphan_dir / "run-000.jsonl", "{\"stale\":true}\n");
    const auto foreign_dir = output_dir / "experiments" / "foreign";
    std::filesystem::create_directories(foreign_dir);
    write_file(foreign_dir / "notes.txt", "keep\n");
    const auto cleanup_result = run_cli({"run", config_path.string()}, work_dir);
    HPOEA_V2_CHECK(runner, cleanup_result.exit_code == 0,
                   "stale cleanup rerun exits successfully");
    HPOEA_V2_CHECK(runner, !std::filesystem::exists(stale_run_path),
                   "rerun removes stale run files the plan no longer produces");
    HPOEA_V2_CHECK(runner, std::filesystem::exists(log_path),
                   "rerun keeps the planned run file");
    HPOEA_V2_CHECK(runner, contains(cleanup_result.stdout_text, "removed stale output"),
                   "rerun reports the removed stale file");
    HPOEA_V2_CHECK(runner, !std::filesystem::exists(orphan_dir),
                   "rerun removes orphan experiment dirs holding only run files");
    HPOEA_V2_CHECK(runner, std::filesystem::exists(foreign_dir / "notes.txt"),
                   "rerun keeps files that are not run outputs");
    HPOEA_V2_CHECK(runner, contains(cleanup_result.stderr_text,
                                    "stale experiment output not in current plan"),
                   "rerun warns about leftover experiment directories");

    {
        // starved run leaves no empty output file
        const auto starved_output_dir = work_dir / "starved-out";
        const auto starved_config_path = work_dir / "starved_run.toml";
        write_file(starved_config_path, starved_pagmo_config(starved_output_dir));

        const auto starved_result = run_cli({"run", starved_config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, starved_result.exit_code == 0,
                       "starved run keeps exit 0 for a budget outcome");
        HPOEA_V2_CHECK(runner, contains(starved_result.stdout_text, "records: 0"),
                       "starved run reports zero records");
        HPOEA_V2_CHECK(runner, contains(starved_result.stdout_text, "status: budget_exceeded"),
                       "starved run reports budget_exceeded status");
        HPOEA_V2_CHECK(runner, contains(starved_result.stderr_text,
                                        "produced no records; removed empty output"),
                       "starved run warns about the removed empty output");
        HPOEA_V2_CHECK(runner, contains(starved_result.stderr_text,
                                        "optimizer budget insufficient"),
                       "starved run prints the optimizer message");
        const auto starved_log_path =
            starved_output_dir / "experiments" / "starved" / "run-000.jsonl";
        HPOEA_V2_CHECK(runner, !std::filesystem::exists(starved_log_path),
                       "starved run leaves no empty JSONL file");
    }

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

    {
        // plan marks both cells runnable
        // run executes them
        const auto matrix_output_dir = work_dir / "matrix-out";
        const auto matrix_config_path = work_dir / "matrix_run.toml";
        write_file(matrix_config_path, matrix_config(matrix_output_dir));

        const auto plan_result = run_cli({"plan", matrix_config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, plan_result.exit_code == 0, "matrix plan exits successfully");
        HPOEA_V2_CHECK(runner, contains(plan_result.stdout_text,
                                        "algorithm: sade_tiny type=sade backend=pagmo dispatch=supported"),
                       "matrix plan marks sade supported");
        HPOEA_V2_CHECK(runner, contains(plan_result.stdout_text,
                                        "optimizer: sa_tiny type=simulated_annealing backend=pagmo dispatch=supported"),
                       "matrix plan marks simulated_annealing supported");
        HPOEA_V2_CHECK(runner, contains(plan_result.stdout_text,
                                        "optimizer: nm_tiny type=nelder_mead backend=pagmo dispatch=supported"),
                       "matrix plan marks nelder_mead supported");
        HPOEA_V2_CHECK(runner, contains(plan_result.stdout_text, "tunable parameters: 1"),
                       "matrix plan shows the sade tunable parameter count");
        HPOEA_V2_CHECK(runner, contains(plan_result.stdout_text, "tunable parameters: 2"),
                       "matrix plan shows the de tunable parameter count");

        const auto matrix_result = run_cli({"run", matrix_config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, matrix_result.exit_code == 0,
                       "matrix run exits successfully");
        HPOEA_V2_CHECK(runner, contains(matrix_result.stdout_text, "ran: rastrigin_sade_sa__rep000"),
                       "matrix run executes the sade cell");
        HPOEA_V2_CHECK(runner, contains(matrix_result.stdout_text, "ran: rosenbrock_de_nm__rep000"),
                       "matrix run executes the nelder_mead cell");

        const auto sade_log = read_file(
            matrix_output_dir / "experiments" / "rastrigin-sade-sa" / "run-000.jsonl");
        HPOEA_V2_CHECK(runner, contains(sade_log, "\"implementation\":\"pagmo::sade\""),
                       "matrix run logs the sade algorithm");
        HPOEA_V2_CHECK(runner, contains(sade_log, "\"implementation\":\"pagmo::simulated_annealing\""),
                       "matrix run logs the simulated_annealing optimizer");
        HPOEA_V2_CHECK(runner, contains(sade_log, "\"problem_id\":\"rastrigin\""),
                       "matrix run logs the rastrigin problem");
        const auto nm_log = read_file(
            matrix_output_dir / "experiments" / "rosenbrock-de-nm" / "run-000.jsonl");
        HPOEA_V2_CHECK(runner, contains(nm_log, "\"implementation\":\"nlopt::neldermead\""),
                       "matrix run logs the nelder_mead optimizer");
    }

    {
        // baseline runs the ea once per repetition
        const auto baseline_output_dir = work_dir / "baseline-out";
        const auto baseline_config_path = work_dir / "baseline_run.toml";
        write_file(baseline_config_path, baseline_config(baseline_output_dir));

        const auto baseline_result = run_cli({"run", baseline_config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, baseline_result.exit_code == 0,
                       "baseline run exits successfully");
        HPOEA_V2_CHECK(runner, contains(baseline_result.stdout_text, "records: 3"),
                       "baseline run writes one tuning and two validation records");
        const auto baseline_log = read_file(
            baseline_output_dir / "experiments" / "baseline" / "run-000.jsonl");
        HPOEA_V2_CHECK(runner, contains(baseline_log, "\"family\":\"Baseline\""),
                       "baseline run logs the baseline optimizer");
        HPOEA_V2_CHECK(runner, contains(baseline_log, "\"implementation\":\"fixed_parameters\""),
                       "baseline run logs the fixed parameter variant");
        HPOEA_V2_CHECK(runner, contains(baseline_log, "\"population_size\":6"),
                       "baseline run applies the fixed parameter");
        HPOEA_V2_CHECK(runner, contains(baseline_log, "\"phase\":\"validation\""),
                       "baseline run writes validation records");
    }

    {
        // degraded run keeps records and must reach stderr
        const auto degraded_output_dir = work_dir / "degraded-out";
        const auto degraded_config_path = work_dir / "degraded_run.toml";
        write_file(degraded_config_path, degraded_config(degraded_output_dir));

        const auto degraded_result = run_cli({"run", degraded_config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, degraded_result.exit_code == 0,
                       "degraded run keeps exit 0 for a budget outcome");
        HPOEA_V2_CHECK(runner, contains(degraded_result.stdout_text, "status: budget_exceeded"),
                       "degraded run reports budget_exceeded status");
        HPOEA_V2_CHECK(runner, contains(degraded_result.stderr_text, "budget_exceeded"),
                       "degraded run reports its status on stderr");
        const auto degraded_log = read_file(
            degraded_output_dir / "experiments" / "degraded" / "run-000.jsonl");
        HPOEA_V2_CHECK(runner, contains(degraded_log, "\"status\":\"budget_exceeded\""),
                       "degraded run keeps its records");
    }

    std::filesystem::remove_all(work_dir);
    return runner.summarize("cli_pagmo_tests");
}
