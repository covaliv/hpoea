#include "cli/dispatch.hpp"

#include "hpoea/config/config_parser.hpp"
#include "hpoea/config/config_validator.hpp"
#include "hpoea/config/suite_expander.hpp"
#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int exit_usage_error = 2;

std::string diagnostic_prefix(std::string_view severity,
                              std::string_view source,
                              std::string_view path) {
    std::string prefix{severity};
    prefix += ": ";
    if (!source.empty()) {
        prefix += source;
        if (!path.empty()) {
            prefix += ':';
        } else {
            prefix += ": ";
        }
    }
    if (!path.empty()) {
        prefix += path;
        prefix += ": ";
    }
    return prefix;
}

void print_help(std::ostream &out) {
    out << "usage: hpoea <command> [args]\n"
        << "\n"
        << "commands:\n"
        << "  validate <config.toml>  validate a config for this build\n"
        << "  plan <config.toml>      preview expanded runs without executing\n"
        << "  run <config.toml>       execute supported config runs\n"
        << "\n"
        << "options:\n"
        << "  --help                  show this help\n"
        << "  --version               show version\n";
}

void print_version(std::ostream &out) {
    out << "hpoea " << HPOEA_PROJECT_VERSION << '\n';
}

int usage_error(std::string_view message) {
    std::cerr << "error: " << message << '\n'
              << "Try 'hpoea --help'.\n";
    return exit_usage_error;
}

void print_parse_diagnostics(const hpoea::config::ParseResult &result) {
    for (const auto &diag : result.diagnostics) {
        const auto severity = diag.severity == hpoea::config::ParseDiagnosticSeverity::Error
            ? "error"
            : "warning";
        std::cerr << diagnostic_prefix(severity, diag.source, diag.path)
                  << diag.message << '\n';
    }
}

void print_validation_diagnostics(const hpoea::config::ValidationResult &result) {
    for (const auto &diag : result.diagnostics) {
        const auto severity = diag.severity == hpoea::config::ValidationDiagnosticSeverity::Error
            ? "error"
            : "warning";
        std::cerr << diagnostic_prefix(severity, {}, diag.path)
                  << diag.message << '\n';
    }
}

void print_expansion_diagnostics(const hpoea::config::ExpansionResult &result) {
    for (const auto &diag : result.diagnostics) {
        const auto severity = diag.severity == hpoea::config::ExpansionDiagnosticSeverity::Error
            ? "error"
            : "warning";
        std::cerr << diagnostic_prefix(severity, {}, diag.path)
                  << diag.message << '\n';
    }
}

std::optional<hpoea::config::SuiteConfig> parse_suite_config(const std::filesystem::path &path) {
    auto result = hpoea::config::parse_config_file(path);
    print_parse_diagnostics(result);
    if (!result.ok() || !result.config.has_value()) {
        if (result.ok()) {
            std::cerr << "error: no suite config was parsed\n";
        }
        return std::nullopt;
    }
    return std::move(*result.config);
}

bool is_backend_unavailable_diagnostic(const hpoea::config::ValidationDiagnostic &diag) {
    return diag.severity == hpoea::config::ValidationDiagnosticSeverity::Error
        && diag.message.find("requires a Pagmo-enabled build") != std::string::npos;
}

bool has_blocking_plan_diagnostics(const hpoea::config::ValidationResult &result) {
    for (const auto &diag : result.diagnostics) {
        if (diag.severity == hpoea::config::ValidationDiagnosticSeverity::Error
            && !is_backend_unavailable_diagnostic(diag)) {
            return true;
        }
    }
    return false;
}

void print_component_line(std::ostream &out,
                          std::string_view label,
                          const hpoea::cli::ComponentDispatch &annotation) {
    out << "  " << label << ": " << annotation.id
        << " type=" << annotation.type
        << " backend=" << annotation.backend
        << " dispatch=" << annotation.dispatch << '\n';
}

hpoea::core::Budget to_core_budget(const hpoea::config::BudgetConfig &budget) {
    hpoea::core::Budget result;
    result.generations = budget.generations;
    result.function_evaluations = budget.function_evaluations;
    return result;
}

bool is_command_failure_status(hpoea::core::RunStatus status) {
    using hpoea::core::RunStatus;
    return status == RunStatus::FailedEvaluation
        || status == RunStatus::InvalidConfiguration
        || status == RunStatus::InternalError;
}

std::optional<unsigned long> to_unsigned_long_seed(std::uint64_t seed) {
    if (seed > static_cast<std::uint64_t>(std::numeric_limits<unsigned long>::max())) {
        return std::nullopt;
    }
    return static_cast<unsigned long>(seed);
}

struct PreparedRun {
    hpoea::config::ResolvedRunSpec run;
    hpoea::cli::DispatchObjects dispatch;
};

int run_validate(const std::filesystem::path &path) {
    const auto config = parse_suite_config(path);
    if (!config.has_value()) {
        return 1;
    }

    const auto result = hpoea::config::validate_suite_config(*config);
    print_validation_diagnostics(result);
    if (!result.ok()) {
        return 1;
    }

    std::cout << "valid: " << path.generic_string() << '\n';
    return 0;
}

int run_plan(const std::filesystem::path &path) {
    const auto config = parse_suite_config(path);
    if (!config.has_value()) {
        return 1;
    }

    const auto validation = hpoea::config::validate_suite_config(*config);
    if (has_blocking_plan_diagnostics(validation)) {
        print_validation_diagnostics(validation);
        return 1;
    }

    const auto expansion = hpoea::config::expand_suite_config(*config);
    print_expansion_diagnostics(expansion);
    if (!expansion.ok()) {
        return 1;
    }

    std::cout << "suite: " << config->name << '\n';
    std::cout << "config: " << path.generic_string() << '\n';
    std::cout << "runs: " << expansion.runs.size() << '\n';

    for (const auto &run : expansion.runs) {
        const auto dispatch = hpoea::cli::annotate_run_dispatch(*config, run);

        std::cout << "run: " << run.run_id << '\n';
        std::cout << "  experiment: " << run.experiment_id << '\n';
        std::cout << "  repetition: " << run.repetition_index << '\n';
        std::cout << "  seed: " << run.seed << '\n';
        std::cout << "  output: " << run.planned_output_path.generic_string() << '\n';
        print_component_line(std::cout, "problem", dispatch.problem);
        print_component_line(std::cout, "algorithm", dispatch.algorithm);
        print_component_line(std::cout, "optimizer", dispatch.optimizer);
        std::cout << "  runnable: " << (dispatch.runnable ? "yes" : "no") << '\n';
    }

    return 0;
}

std::optional<hpoea::core::ExperimentConfig> make_experiment_config(
    const hpoea::config::SuiteConfig &suite,
    const hpoea::config::ResolvedRunSpec &run) {
    const auto *algorithm = hpoea::cli::find_algorithm(suite, run.algorithm_id);
    const auto *optimizer = hpoea::cli::find_optimizer(suite, run.optimizer_id);
    if (!algorithm) {
        std::cerr << "error: run " << run.run_id << ": missing algorithm id: "
                  << run.algorithm_id << '\n';
        return std::nullopt;
    }
    if (!optimizer) {
        std::cerr << "error: run " << run.run_id << ": missing optimizer id: "
                  << run.optimizer_id << '\n';
        return std::nullopt;
    }

    const auto seed = to_unsigned_long_seed(run.seed);
    if (!seed.has_value()) {
        std::cerr << "error: run " << run.run_id << ": seed is too large for this platform: "
                  << run.seed << '\n';
        return std::nullopt;
    }

    hpoea::core::ExperimentConfig config;
    config.experiment_id = run.run_id;
    config.trials_per_optimizer = 1;
    config.islands = 1;
    config.algorithm_budget = to_core_budget(run.algorithm_budget);
    config.optimizer_budget = to_core_budget(run.optimizer_budget);
    config.log_file_path = run.planned_output_path;
    config.random_seed = *seed;
    if (!algorithm->fixed_parameters.empty()) {
        config.algorithm_baseline_parameters = algorithm->fixed_parameters;
    }
    if (!optimizer->parameters.empty()) {
        config.optimizer_parameters = optimizer->parameters;
    }
    return config;
}

bool create_parent_directory(const std::filesystem::path &path) {
    const auto parent = path.parent_path();
    if (parent.empty()) {
        return true;
    }
    std::filesystem::create_directories(parent);
    return true;
}

int run_config(const std::filesystem::path &path) {
    const auto config = parse_suite_config(path);
    if (!config.has_value()) {
        return 1;
    }

    const auto validation = hpoea::config::validate_suite_config(*config);
    print_validation_diagnostics(validation);
    if (!validation.ok()) {
        return 1;
    }

    // suite was already expanded and reported above
    // only fetch run plans here
    const auto expansion = hpoea::config::expand_suite_config(*config);
    if (!expansion.ok()) {
        return 1;
    }

    std::vector<PreparedRun> prepared_runs;
    prepared_runs.reserve(expansion.runs.size());
    bool dispatch_ok = true;
    for (const auto &run : expansion.runs) {
        auto dispatch = hpoea::cli::make_dispatch_objects(*config, run);
        if (!dispatch.ok()) {
            dispatch_ok = false;
            for (const auto &error : dispatch.errors) {
                std::cerr << "error: run " << run.run_id << ": " << error << '\n';
            }
            continue;
        }
        prepared_runs.push_back(PreparedRun{run, std::move(dispatch.objects)});
    }
    if (!dispatch_ok) {
        return 1;
    }

    bool had_failed_status = false;
    try {
        for (auto &prepared : prepared_runs) {
            auto experiment_config = make_experiment_config(*config, prepared.run);
            if (!experiment_config.has_value()) {
                return 1;
            }

            create_parent_directory(experiment_config->log_file_path);
            // logger appends so a rerun must clear old records first
            if (std::filesystem::exists(experiment_config->log_file_path)) {
                std::filesystem::remove(experiment_config->log_file_path);
            }
            hpoea::core::JsonlLogger logger{experiment_config->log_file_path};
            hpoea::core::SequentialExperimentManager manager;
            const auto result = manager.run_experiment(
                *experiment_config,
                *prepared.dispatch.optimizer,
                *prepared.dispatch.algorithm_factory,
                *prepared.dispatch.problem,
                logger);

            std::cout << "ran: " << prepared.run.run_id << '\n';
            std::cout << "  output: " << experiment_config->log_file_path.generic_string() << '\n';
            std::cout << "  optimizer_runs: " << result.optimizer_results.size() << '\n';
            std::cout << "  records: " << logger.records_written() << '\n';
            for (const auto &optimizer_result : result.optimizer_results) {
                const auto status = hpoea::core::detail::run_status_to_string(optimizer_result.status);
                std::cout << "  status: " << status << '\n';
                if (is_command_failure_status(optimizer_result.status)) {
                    had_failed_status = true;
                }
            }
        }
    } catch (const std::exception &exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }

    return had_failed_status ? 1 : 0;
}

int require_single_config_arg(int argc,
                              char **argv,
                              int (*handler)(const std::filesystem::path &)) {
    if (argc < 3) {
        return usage_error("missing config path");
    }
    if (argc > 3) {
        return usage_error("too many arguments for command: " + std::string{argv[1]});
    }
    return handler(std::filesystem::path{argv[2]});
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 1) {
        return usage_error("missing command");
    }

    const std::string_view command{argv[1]};
    if (command == "--help") {
        print_help(std::cout);
        return 0;
    }
    if (command == "--version") {
        print_version(std::cout);
        return 0;
    }
    if (command == "validate") {
        return require_single_config_arg(argc, argv, run_validate);
    }
    if (command == "plan") {
        return require_single_config_arg(argc, argv, run_plan);
    }
    if (command == "run") {
        return require_single_config_arg(argc, argv, run_config);
    }

    return usage_error("unknown command: " + std::string{command});
}
