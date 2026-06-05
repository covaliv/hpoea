#include "test_harness.hpp"

#include <array>
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

std::filesystem::path project_path(std::initializer_list<std::string_view> parts) {
    std::filesystem::path path{HPOEA_PROJECT_SOURCE_DIR};
    for (const auto part : parts) {
        path /= part;
    }
    return path;
}

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

std::string read_basic_example_with_output_dir(const std::filesystem::path &output_dir) {
    auto text = read_file(project_path({"examples", "configs", "basic_experiment.toml"}));
    const std::string from = "output_dir = \"results/basic_experiment\"";
    const std::string to = "output_dir = \"" + output_dir.generic_string() + "\"";
    const auto position = text.find(from);
    if (position != std::string::npos) {
        text.replace(position, from.size(), to);
    }
    return text;
}

std::string custom_ids_config_with_output_dir(const std::filesystem::path &output_dir) {
    return R"(schema_version = 1

[suite]
name = "custom_ids"
output_dir = ")" + output_dir.generic_string() + R"("
suite_seed = 77
repetitions = 1

[problems.local_problem]
type = "local_problem"

[algorithms.local_algorithm]
type = "local_algorithm"
fixed = { population_size = 12 }

[optimizers.local_optimizer]
type = "local_optimizer"
parameters = { generations = 1 }

[[experiments]]
id = "custom_id_run"
problem = "local_problem"
algorithm = "local_algorithm"
optimizer = "local_optimizer"
seed = 11

[experiments.algorithm_budget]
generations = 1

[experiments.optimizer_budget]
function_evaluations = 1
)";
}

std::string random_search_config_with_output_dir(const std::filesystem::path &output_dir) {
    return R"(schema_version = 1

[suite]
name = "random_search_cli"
output_dir = ")" + output_dir.generic_string() + R"("
suite_seed = 77
repetitions = 1

[problems.sphere]
type = "sphere"
dimension = 2

[algorithms.de_default]
type = "de"

[optimizers.rs]
type = "random_search"
parameters = { sample_count = 2 }

[[experiments]]
id = "sphere_de_random"
problem = "sphere"
algorithm = "de_default"
optimizer = "rs"
seed = 11

[experiments.algorithm_budget]
generations = 1

[experiments.optimizer_budget]
function_evaluations = 2
)";
}

bool contains(const std::string &text,
              const std::string &needle) {
    return text.find(needle) != std::string::npos;
}

} // namespace

int main() {
    hpoea::tests_v2::TestRunner runner;
    const auto work_dir = unique_test_dir("cli_tests");
    std::filesystem::remove_all(work_dir);
    std::filesystem::create_directories(work_dir);

    {
        const auto result = run_cli({"--help"}, work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code == 0, "help exits successfully");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text, "usage: hpoea"),
                       "help prints usage");
        HPOEA_V2_CHECK(runner, result.stderr_text.empty(), "help does not print stderr");
    }

    {
        const auto result = run_cli({"--version"}, work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code == 0, "version exits successfully");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text, std::string{"hpoea "} + HPOEA_PROJECT_VERSION),
                       "version prints project version");
        HPOEA_V2_CHECK(runner, result.stderr_text.empty(), "version does not print stderr");
    }

    {
        const auto result = run_cli({"__hpoea_unknown_command__"}, work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code != 0, "unknown command exits nonzero");
        HPOEA_V2_CHECK(runner, result.stdout_text.empty(), "unknown command does not print stdout");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text, "unknown command"),
                       "unknown command reports diagnostic");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text, "--help"),
                       "unknown command includes help hint");
    }

    {
        const auto result = run_cli(
            {"validate", project_path({"tests", "fixtures", "configs", "custom_ids_valid.toml"}).string()},
            work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code == 0, "custom id config validates");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text, "valid:"),
                       "validate success prints valid marker");
        HPOEA_V2_CHECK(runner, result.stderr_text.empty(), "custom id validation has no stderr");
    }

    {
        const auto output_dir = work_dir / "custom-run-output";
        const auto config_path = work_dir / "custom_ids_run.toml";
        write_file(config_path, custom_ids_config_with_output_dir(output_dir));
        std::filesystem::remove_all(output_dir);

        const auto result = run_cli({"run", config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code != 0,
                       "valid custom id config fails unsupported dispatch");
        HPOEA_V2_CHECK(runner, result.stdout_text.empty(),
                       "unsupported dispatch failure has no stdout");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text,
                                       "unsupported dispatch for problem type: local_problem"),
                       "unsupported dispatch reports custom problem");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text,
                                       "unsupported dispatch for algorithm type: local_algorithm"),
                       "unsupported dispatch reports custom algorithm");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text,
                                       "unsupported dispatch for optimizer type: local_optimizer"),
                       "unsupported dispatch reports custom optimizer");
        HPOEA_V2_CHECK(runner, !std::filesystem::exists(output_dir),
                       "unsupported dispatch failure does not create output directory");
    }

    {
        const auto result = run_cli(
            {"validate", project_path({"examples", "configs", "basic_experiment.toml"}).string()},
            work_dir);
#if defined(HPOEA_CONFIG_HAS_PAGMO)
        HPOEA_V2_CHECK(runner, result.exit_code == 0,
                       "Pagmo build validates basic example");
#else
        HPOEA_V2_CHECK(runner, result.exit_code != 0,
                       "core-only basic example validation fails without Pagmo");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text,
                                       "algorithm type requires a Pagmo-enabled build: de"),
                       "validate reports unavailable de");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text,
                                       "optimizer type requires a Pagmo-enabled build: cmaes"),
                       "validate reports unavailable cmaes");
#endif
    }

    {
        const auto output_dir = work_dir / "random-search-plan-output";
        const auto config_path = work_dir / "random_search_plan.toml";
        write_file(config_path, random_search_config_with_output_dir(output_dir));
        std::filesystem::remove_all(output_dir);

        const auto result = run_cli({"plan", config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code == 0,
                       "plan previews random_search config");
        HPOEA_V2_CHECK(runner, result.stderr_text.empty(),
                       "random_search plan preview has no stderr");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text,
                                       "optimizer: rs type=random_search backend=core dispatch=supported"),
                       "plan preview annotates random_search as core supported");
#if defined(HPOEA_CONFIG_HAS_PAGMO)
        HPOEA_V2_CHECK(runner, contains(result.stdout_text, "runnable: yes"),
                       "Pagmo plan preview marks random_search run runnable with de");
#else
        HPOEA_V2_CHECK(runner, contains(result.stdout_text,
                                       "algorithm: de_default type=de backend=pagmo-unavailable dispatch=requires-pagmo"),
                       "core-only random_search plan still reports unavailable de");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text, "runnable: no"),
                       "core-only random_search plan is not runnable without de backend");
#endif
        HPOEA_V2_CHECK(runner, !std::filesystem::exists(output_dir),
                       "random_search plan preview does not create output directory");
    }

#if !defined(HPOEA_CONFIG_HAS_PAGMO)
    {
        const auto output_dir = work_dir / "random-search-run-output";
        const auto config_path = work_dir / "random_search_run.toml";
        write_file(config_path, random_search_config_with_output_dir(output_dir));
        std::filesystem::remove_all(output_dir);

        const auto result = run_cli({"run", config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code != 0,
                       "core-only random_search run fails without de backend");
        HPOEA_V2_CHECK(runner, result.stdout_text.empty(),
                       "core-only random_search run failure has no stdout");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text,
                                       "algorithm type requires a Pagmo-enabled build: de"),
                       "core-only random_search run reports unavailable de");
        HPOEA_V2_CHECK(runner, !contains(result.stderr_text,
                                        "unsupported dispatch for optimizer type: random_search"),
                       "core-only random_search run does not reject random_search optimizer");
        HPOEA_V2_CHECK(runner, !std::filesystem::exists(output_dir),
                       "core-only random_search run does not create output directory");
    }
#endif

    {
        const auto output_dir = work_dir / "plan-output";
        const auto config_path = work_dir / "basic_plan.toml";
        write_file(config_path, read_basic_example_with_output_dir(output_dir));
        std::filesystem::remove_all(output_dir);

        const auto result = run_cli({"plan", config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code == 0,
                       "core-only plan previews basic example");
        HPOEA_V2_CHECK(runner, result.stderr_text.empty(), "plan preview has no stderr");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text, "sphere_de_cmaes__rep000")
                                  && contains(result.stdout_text, "sphere_de_cmaes__rep001")
                                  && contains(result.stdout_text, "sphere_de_cmaes__rep002"),
                       "plan preview lists all repetitions");
#if defined(HPOEA_CONFIG_HAS_PAGMO)
        HPOEA_V2_CHECK(runner, contains(result.stdout_text,
                                       "algorithm: de_default type=de backend=pagmo dispatch=supported"),
                       "plan preview annotates supported de dispatch");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text,
                                       "optimizer: cmaes_fast type=cmaes backend=pagmo dispatch=supported"),
                       "plan preview annotates supported cmaes dispatch");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text, "runnable: yes"),
                       "Pagmo plan preview marks canonical run runnable");
#else
        HPOEA_V2_CHECK(runner, contains(result.stdout_text,
                                       "algorithm: de_default type=de backend=pagmo-unavailable dispatch=requires-pagmo"),
                       "plan preview annotates unavailable de dispatch");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text,
                                       "optimizer: cmaes_fast type=cmaes backend=pagmo-unavailable dispatch=requires-pagmo"),
                       "plan preview annotates unavailable cmaes dispatch");
        HPOEA_V2_CHECK(runner, contains(result.stdout_text, "runnable: no"),
                       "core-only plan preview marks canonical run not runnable");
#endif
        HPOEA_V2_CHECK(runner, !std::filesystem::exists(output_dir),
                       "plan preview does not create output directory");
    }

#if !defined(HPOEA_CONFIG_HAS_PAGMO)
    {
        const auto output_dir = work_dir / "run-output";
        const auto config_path = work_dir / "basic_run.toml";
        write_file(config_path, read_basic_example_with_output_dir(output_dir));
        std::filesystem::remove_all(output_dir);

        const auto result = run_cli({"run", config_path.string()}, work_dir);
        HPOEA_V2_CHECK(runner, result.exit_code != 0,
                       "core-only run fails before execution");
        HPOEA_V2_CHECK(runner, result.stdout_text.empty(),
                       "core-only run failure has no stdout");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text,
                                       "algorithm type requires a Pagmo-enabled build: de"),
                       "run reports unavailable de");
        HPOEA_V2_CHECK(runner, contains(result.stderr_text,
                                       "optimizer type requires a Pagmo-enabled build: cmaes"),
                       "run reports unavailable cmaes");
        HPOEA_V2_CHECK(runner, !std::filesystem::exists(output_dir),
                       "core-only run failure does not create output directory");
    }
#endif

    std::filesystem::remove_all(work_dir);
    return runner.summarize("cli_tests");
}
