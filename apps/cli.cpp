#include "cli/dispatch.hpp"

#include "hpoea/config/config_parser.hpp"
#include "hpoea/config/config_validator.hpp"
#include "hpoea/config/suite_expander.hpp"
#include "hpoea/core/experiment.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/logging.hpp"

#include <cctype>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
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
        << "run options:\n"
        << "  --only <id[,id...]>     run only the named experiments\n"
        << "  --prune                 remove experiment outputs no longer in the plan\n"
        << "  --resume                skip experiments whose output already exists\n"
        << "  --strict                exit nonzero when a cell is degraded or empty\n"
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

    bool plan_ok = true;
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
        if (const auto *algorithm = hpoea::cli::find_algorithm(*config, run.algorithm_id)) {
            if (const auto tunable = hpoea::cli::tuned_algorithm_dimension(*algorithm)) {
                std::cout << "  tunable parameters: " << *tunable << '\n';
            }
        }
        bool runnable = dispatch.runnable;
        if (runnable) {
            auto objects = hpoea::cli::make_dispatch_objects(*config, run);
            if (!objects.ok()) {
                runnable = false;
                plan_ok = false;
                for (const auto &error : objects.errors) {
                    std::cerr << "error: run " << run.run_id << ": " << error << '\n';
                }
            }
        }
        std::cout << "  runnable: " << (runnable ? "yes" : "no") << '\n';
    }

    return plan_ok ? 0 : 1;
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
    config.max_parallel_trials = 1;
    config.validation_repeats = run.validation_repeats;
    config.algorithm_budget = to_core_budget(run.algorithm_budget);
    config.optimizer_budget = to_core_budget(run.optimizer_budget);
    config.log_file_path = run.planned_output_path;
    config.random_seed = *seed;
    // baseline applies fixed parameters itself
    if (optimizer->type != "baseline" && !algorithm->fixed_parameters.empty()) {
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

bool is_run_output_file(const std::filesystem::path &path) {
    const auto name = path.filename().string();
    constexpr std::string_view prefix{"run-"};
    constexpr std::string_view suffix{".jsonl"};
    if (name.size() <= prefix.size() + suffix.size()) {
        return false;
    }
    if (name.compare(0, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }
    for (std::size_t i = prefix.size(); i < name.size() - suffix.size(); ++i) {
        if (std::isdigit(static_cast<unsigned char>(name[i])) == 0) {
            return false;
        }
    }
    return true;
}

struct RunOptions {
    std::set<std::string> only;
    bool prune{false};
    bool strict{false};
    bool resume{false};
};

struct RunTally {
    std::size_t ran{0};
    std::size_t skipped{0};
    std::size_t success{0};
    std::size_t budget_exceeded{0};
    std::size_t failed{0};
    std::size_t empty{0};
    std::size_t no_selectable{0};
};

std::string json_escape(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    return out;
}

std::string manifest_row(const hpoea::config::ResolvedRunSpec &run,
                         std::string_view status,
                         std::size_t records,
                         std::size_t selectable) {
    std::ostringstream oss;
    oss << '{'
        << "\"run_id\":\"" << json_escape(run.run_id) << "\","
        << "\"experiment_id\":\"" << json_escape(run.experiment_id) << "\","
        << "\"problem_id\":\"" << json_escape(run.problem_id) << "\","
        << "\"algorithm_id\":\"" << json_escape(run.algorithm_id) << "\","
        << "\"optimizer_id\":\"" << json_escape(run.optimizer_id) << "\","
        << "\"repetition\":" << run.repetition_index << ','
        << "\"output\":\"" << json_escape(run.planned_output_path.generic_string()) << "\","
        << "\"status\":\"" << json_escape(status) << "\","
        << "\"records\":" << records << ','
        << "\"selectable\":" << selectable
        << '}';
    return oss.str();
}

// run_id is always first field of a self written row
std::string manifest_run_id(const std::string &row) {
    constexpr std::string_view prefix{"{\"run_id\":\""};
    if (!row.starts_with(prefix)) {
        return {};
    }
    std::string id;
    for (auto i = prefix.size(); i < row.size(); ++i) {
        if (row[i] == '\\' && i + 1 < row.size()) {
            id += row[++i];
        } else if (row[i] == '"') {
            return id;
        } else {
            id += row[i];
        }
    }
    return {};
}

// partial run keeps rows of cells it did not touch
// rows leave manifest only when they leave plan
void write_run_manifest(const std::filesystem::path &output_dir,
                        const std::vector<hpoea::config::ResolvedRunSpec> &planned_runs,
                        const std::map<std::string, std::string> &fresh_rows,
                        const std::map<std::string, std::string> &skip_rows) {
    if (fresh_rows.empty() && skip_rows.empty()) {
        return;
    }
    const auto manifest_path = output_dir / "summary.jsonl";
    std::map<std::string, std::string> rows;
    {
        std::ifstream existing(manifest_path);
        std::string line;
        while (std::getline(existing, line)) {
            if (auto id = manifest_run_id(line); !id.empty()) {
                rows.insert_or_assign(std::move(id), line);
            }
        }
    }
    for (const auto &[id, row] : skip_rows) {
        // previous row describes untouched output better
        rows.emplace(id, row);
    }
    for (const auto &[id, row] : fresh_rows) {
        rows.insert_or_assign(id, row);
    }
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    std::ofstream stream(manifest_path, std::ios::out | std::ios::trunc);
    if (!stream) {
        std::cerr << "warning: could not write run manifest under " << output_dir.generic_string() << '\n';
        return;
    }
    for (const auto &run : planned_runs) {
        const auto row = rows.find(run.run_id);
        if (row != rows.end()) {
            stream << row->second << '\n';
        }
    }
}

// a config change can leave stale run files behind
void remove_stale_outputs(const hpoea::config::SuiteConfig &config,
                               const std::vector<PreparedRun> &prepared_runs,
                               const RunOptions &options) {
    std::set<std::filesystem::path> planned_files;
    std::set<std::filesystem::path> planned_dirs;
    for (const auto &prepared : prepared_runs) {
        auto planned = prepared.run.planned_output_path.lexically_normal();
        planned_dirs.insert(planned.parent_path());
        planned_files.insert(std::move(planned));
    }

    for (const auto &dir : planned_dirs) {
        if (!std::filesystem::is_directory(dir)) {
            continue;
        }
        for (const auto &entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file() || !is_run_output_file(entry.path())) {
                continue;
            }
            if (planned_files.contains(entry.path().lexically_normal())) {
                continue;
            }
            std::filesystem::remove(entry.path());
            std::cout << "removed stale output: " << entry.path().generic_string() << '\n';
        }
    }

    if (!options.only.empty()) {
        return;
    }

    const auto experiments_root = (config.output_dir / "experiments").lexically_normal();
    if (!std::filesystem::is_directory(experiments_root)) {
        return;
    }
    for (const auto &entry : std::filesystem::directory_iterator(experiments_root)) {
        if (!entry.is_directory() || planned_dirs.contains(entry.path().lexically_normal())) {
            continue;
        }
        if (!options.prune) {
            std::cerr << "warning: stale experiment output not in current plan: "
                      << entry.path().generic_string() << " (use --prune to remove)\n";
            continue;
        }
        for (const auto &stale : std::filesystem::directory_iterator(entry.path())) {
            if (!stale.is_regular_file() || !is_run_output_file(stale.path())) {
                continue;
            }
            std::filesystem::remove(stale.path());
            std::cout << "removed stale output: " << stale.path().generic_string() << '\n';
        }
        if (std::filesystem::is_empty(entry.path())) {
            std::filesystem::remove(entry.path());
            continue;
        }
        std::cerr << "warning: stale experiment output not in current plan: "
                  << entry.path().generic_string() << '\n';
    }
}

int run_config(const std::filesystem::path &path, const RunOptions &options) {
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

    if (!options.only.empty()) {
        std::set<std::string> present;
        for (const auto &run : expansion.runs) {
            present.insert(run.experiment_id);
        }
        for (const auto &id : options.only) {
            if (!present.contains(id)) {
                std::cerr << "error: --only names unknown experiment: " << id << '\n';
                return 1;
            }
        }
    }

    std::vector<PreparedRun> prepared_runs;
    prepared_runs.reserve(expansion.runs.size());
    bool dispatch_ok = true;
    for (const auto &run : expansion.runs) {
        if (!options.only.empty() && !options.only.contains(run.experiment_id)) {
            continue;
        }
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

    try {
        remove_stale_outputs(*config, prepared_runs, options);
    } catch (const std::exception &exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }

    RunTally tally;
    std::map<std::string, std::string> fresh_rows;
    std::map<std::string, std::string> skip_rows;

    bool had_failed_status = false;
    try {
        for (auto &prepared : prepared_runs) {
            const auto &run = prepared.run;

            if (options.resume && std::filesystem::exists(run.planned_output_path) &&
                !std::filesystem::is_empty(run.planned_output_path)) {
                std::cout << "skipped: " << run.run_id << " (resume; output exists)\n";
                tally.skipped += 1;
                skip_rows.emplace(run.run_id, manifest_row(run, "skipped", 0, 0));
                continue;
            }

            auto experiment_config = make_experiment_config(*config, run);
            if (!experiment_config.has_value()) {
                return 1;
            }

            create_parent_directory(experiment_config->log_file_path);
            // logger appends
            // so a rerun must clear old records first
            if (std::filesystem::exists(experiment_config->log_file_path)) {
                std::filesystem::remove(experiment_config->log_file_path);
            }
            hpoea::core::SequentialExperimentManager manager;
            hpoea::core::ExperimentResult result;
            std::size_t records_written = 0;
            {
                hpoea::core::JsonlLogger logger{experiment_config->log_file_path};
                result = manager.run_experiment(
                    *experiment_config,
                    *prepared.dispatch.optimizer,
                    *prepared.dispatch.algorithm_factory,
                    *prepared.dispatch.problem,
                    logger);
                records_written = logger.records_written();
            }
            tally.ran += 1;

            std::cout << "ran: " << run.run_id << '\n';
            std::cout << "  output: " << experiment_config->log_file_path.generic_string() << '\n';
            std::cout << "  optimizer_runs: " << result.optimizer_results.size() << '\n';
            std::cout << "  records: " << records_written << '\n';

            hpoea::core::RunStatus cell_status = hpoea::core::RunStatus::InternalError;
            std::size_t selectable = 0;
            for (const auto &optimizer_result : result.optimizer_results) {
                cell_status = optimizer_result.status;
                std::cout << "  status: " << hpoea::core::detail::run_status_to_string(optimizer_result.status) << '\n';
                if (is_command_failure_status(optimizer_result.status)) {
                    had_failed_status = true;
                }
                for (const auto &trial : optimizer_result.trials) {
                    if (hpoea::core::is_selectable_trial(trial)) {
                        selectable += 1;
                    }
                }
            }

            if (cell_status == hpoea::core::RunStatus::Success) {
                tally.success += 1;
            } else if (cell_status == hpoea::core::RunStatus::BudgetExceeded) {
                tally.budget_exceeded += 1;
            } else {
                tally.failed += 1;
            }
            if (selectable == 0) {
                tally.no_selectable += 1;
            }
            fresh_rows.insert_or_assign(run.run_id, manifest_row(
                run, hpoea::core::detail::run_status_to_string(cell_status), records_written, selectable));

            if (records_written == 0) {
                tally.empty += 1;
                std::filesystem::remove(experiment_config->log_file_path);
                std::cerr << "warning: run " << run.run_id
                          << " produced no records; removed empty output "
                          << experiment_config->log_file_path.generic_string() << '\n';
                for (const auto &optimizer_result : result.optimizer_results) {
                    if (!optimizer_result.message.empty()) {
                        std::cerr << "warning: run " << run.run_id << ": "
                                  << optimizer_result.message << '\n';
                    }
                }
            } else {
                // exit code stays 0 for budget outcomes
                // so degraded runs must at least reach stderr
                for (const auto &optimizer_result : result.optimizer_results) {
                    if (optimizer_result.status != hpoea::core::RunStatus::Success) {
                        std::cerr << "warning: run " << run.run_id << ": "
                                  << hpoea::core::detail::run_status_to_string(optimizer_result.status);
                        if (!optimizer_result.message.empty()) {
                            std::cerr << ": " << optimizer_result.message;
                        }
                        std::cerr << '\n';
                    }
                    std::size_t failed_validations = 0;
                    for (const auto &validation_run : optimizer_result.validation_runs) {
                        if (is_command_failure_status(validation_run.status)) {
                            failed_validations += 1;
                        }
                    }
                    if (failed_validations > 0) {
                        std::cerr << "warning: run " << run.run_id
                                  << ": failed validation runs: " << failed_validations << '\n';
                    }
                }
            }
        }
    } catch (const std::exception &exception) {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }

    write_run_manifest(config->output_dir, expansion.runs, fresh_rows, skip_rows);

    std::cout << "suite summary:\n";
    std::cout << "  cells: " << prepared_runs.size() << "  ran: " << tally.ran
              << "  skipped: " << tally.skipped << '\n';
    std::cout << "  success: " << tally.success << "  budget_exceeded: " << tally.budget_exceeded
              << "  failed: " << tally.failed << '\n';
    if (tally.empty > 0 || tally.no_selectable > 0) {
        std::cout << "  empty: " << tally.empty << "  no selectable best: " << tally.no_selectable << '\n';
    }

    if (had_failed_status) {
        return 1;
    }
    if (options.strict && (tally.budget_exceeded > 0 || tally.empty > 0 || tally.no_selectable > 0)) {
        return 1;
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

void split_experiment_ids(std::string_view text, std::set<std::string> &out) {
    std::istringstream ids{std::string{text}};
    std::string id;
    while (std::getline(ids, id, ',')) {
        if (!id.empty()) {
            out.insert(id);
        }
    }
}

int parse_run_options(int argc, char **argv, std::filesystem::path &path, RunOptions &options) {
    bool have_path = false;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--prune") {
            options.prune = true;
        } else if (arg == "--strict") {
            options.strict = true;
        } else if (arg == "--resume") {
            options.resume = true;
        } else if (arg == "--only") {
            if (i + 1 >= argc) {
                return usage_error("--only needs a comma-separated list of experiment ids");
            }
            split_experiment_ids(argv[++i], options.only);
        } else if (arg.starts_with("--only=")) {
            split_experiment_ids(arg.substr(7), options.only);
        } else if (arg.starts_with('-')) {
            return usage_error("unknown option for run: " + std::string{arg});
        } else if (!have_path) {
            path = std::filesystem::path{arg};
            have_path = true;
        } else {
            return usage_error("too many arguments for command: run");
        }
    }
    if (!have_path) {
        return usage_error("missing config path");
    }
    return 0;
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
        std::filesystem::path path;
        RunOptions options;
        if (const auto parsed = parse_run_options(argc, argv, path, options); parsed != 0) {
            return parsed;
        }
        return run_config(path, options);
    }

    return usage_error("unknown command: " + std::string{command});
}
