#include "hpoea/config/config_parser.hpp"
#include "hpoea/config/config_validator.hpp"
#include "hpoea/config/suite_expander.hpp"

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

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

int command_not_ready(std::string_view command) {
    std::cerr << "error: command is not implemented yet: " << command << '\n';
    return 1;
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
        std::cout << "run: " << run.run_id << '\n';
        std::cout << "  experiment: " << run.experiment_id << '\n';
        std::cout << "  repetition: " << run.repetition_index << '\n';
        std::cout << "  seed: " << run.seed << '\n';
        std::cout << "  output: " << run.planned_output_path.generic_string() << '\n';
    }

    return 0;
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
        return command_not_ready(command);
    }

    return usage_error("unknown command: " + std::string{command});
}
