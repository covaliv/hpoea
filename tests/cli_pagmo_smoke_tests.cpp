#include "test_harness.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

struct CommandResult {
    int exit_code{0};
    std::string stdout_text;
    std::string stderr_text;
};

std::filesystem::path unique_test_dir(const std::string &name) {
    return std::filesystem::temp_directory_path()
        / (name + "_" + std::to_string(::getpid()));
}

std::string shell_quote(const std::string &value) {
    std::string quoted = "'";
    for (const auto ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream stream{path};
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

void write_file(const std::filesystem::path &path,
                const std::string &text) {
    std::ofstream stream{path};
    stream << text;
}

CommandResult run_cli(const std::vector<std::string> &args,
                      const std::filesystem::path &work_dir) {
    static int command_index = 0;
    std::filesystem::create_directories(work_dir);
    const auto stdout_path = work_dir / ("stdout_" + std::to_string(command_index) + ".txt");
    const auto stderr_path = work_dir / ("stderr_" + std::to_string(command_index) + ".txt");
    ++command_index;

    std::string command = shell_quote(CLI_EXECUTABLE);
    for (const auto &arg : args) {
        command += ' ';
        command += shell_quote(arg);
    }
    command += " > ";
    command += shell_quote(stdout_path.string());
    command += " 2> ";
    command += shell_quote(stderr_path.string());

    const int raw_status = std::system(command.c_str());
    CommandResult result;
    if (raw_status == -1) {
        result.exit_code = errno == 0 ? 1 : errno;
    } else if (WIFEXITED(raw_status)) {
        result.exit_code = WEXITSTATUS(raw_status);
    } else {
        result.exit_code = 1;
    }
    result.stdout_text = read_file(stdout_path);
    result.stderr_text = read_file(stderr_path);
    return result;
}

bool contains(const std::string &text,
              const std::string &needle) {
    return text.find(needle) != std::string::npos;
}

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
