#include "hpoea/config/suite_expander.hpp"


#include <cctype>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {

using hpoea::config::BudgetConfig;
using hpoea::config::ExpansionDiagnostic;
using hpoea::config::ExpansionDiagnosticSeverity;
using hpoea::config::ExpansionResult;
using hpoea::config::ExperimentSpec;
using hpoea::config::ResolvedRunSpec;
using hpoea::config::SuiteConfig;

std::string join_path(std::string_view base,
                      std::string_view key) {
    if (base.empty()) {
        return std::string{key};
    }
    std::string path{base};
    path += '.';
    path += key;
    return path;
}

std::string join_index(std::string_view base,
                       std::size_t index) {
    std::string path{base};
    path += '[';
    path += std::to_string(index);
    path += ']';
    return path;
}

std::string format_repetition_index(std::size_t value) {
    std::ostringstream stream;
    stream << std::setw(3) << std::setfill('0') << value;
    return stream.str();
}

std::filesystem::path make_output_path(const std::filesystem::path &output_dir,
                                       const std::string &output_name,
                                       std::size_t repetition_index) {
    std::string file_name = "run-";
    file_name += format_repetition_index(repetition_index);
    file_name += ".jsonl";
    return output_dir / "experiments" / output_name / file_name;
}

std::uint64_t fnv1a_append(std::uint64_t state,
                           std::string_view text) {
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (const unsigned char ch : text) {
        state ^= static_cast<std::uint64_t>(ch);
        state *= prime;
    }
    state ^= 0xffU;
    state *= prime;
    return state;
}

std::uint64_t derive_hashed_seed(const SuiteConfig &cfg,
                                 const ExperimentSpec &exp,
                                 std::size_t repetition_index) {
    std::uint64_t state = 14695981039346656037ULL;
    state = fnv1a_append(state, "suite_seed");
    state = fnv1a_append(state, cfg.suite_seed ? std::to_string(*cfg.suite_seed) : std::string{"0"});
    state = fnv1a_append(state, "experiment_id");
    state = fnv1a_append(state, exp.id);
    state = fnv1a_append(state, "problem_id");
    state = fnv1a_append(state, exp.problem);
    state = fnv1a_append(state, "algorithm_id");
    state = fnv1a_append(state, exp.algorithm);
    state = fnv1a_append(state, "optimizer_id");
    state = fnv1a_append(state, exp.optimizer);
    state = fnv1a_append(state, "repetition_index");
    state = fnv1a_append(state, std::to_string(repetition_index));
    return state;
}

std::optional<std::uint64_t> derive_seed(const SuiteConfig &cfg,
                                         const ExperimentSpec &exp,
                                         std::size_t repetition_index) {
    if (!exp.seed.has_value()) {
        return derive_hashed_seed(cfg, exp, repetition_index);
    }

    const auto offset = static_cast<std::uint64_t>(repetition_index);
    if (static_cast<std::size_t>(offset) != repetition_index
        || *exp.seed > std::numeric_limits<std::uint64_t>::max() - offset) {
        return std::nullopt;
    }
    return *exp.seed + offset;
}

BudgetConfig budget_or_empty(const std::optional<BudgetConfig> &budget) {
    return budget.value_or(BudgetConfig{});
}

class Expander {
public:
    explicit Expander(const SuiteConfig &config)
        : config_(config) {}

    ExpansionResult expand() {
        expand_experiments();
        if (result_.has_errors()) {
            result_.runs.clear();
        }
        return result_;
    }

private:
    void add_diagnostic(ExpansionDiagnosticSeverity severity,
                        std::string path,
                        std::string message) {
        result_.diagnostics.push_back(ExpansionDiagnostic{severity, std::move(path), std::move(message)});
    }

    void add_error(std::string path,
                   std::string message) {
        add_diagnostic(ExpansionDiagnosticSeverity::Error, std::move(path), std::move(message));
    }

    std::optional<std::string> validate_output_name(std::string_view raw,
                                                    std::string_view path,
                                                    std::string_view label) {
        if (raw.empty()) {
            add_error(std::string{path}, std::string{label} + " must not be empty");
            return std::nullopt;
        }
        if (raw == "." || raw == "..") {
            add_error(std::string{path}, std::string{label} + " must not be '.' or '..'");
            return std::nullopt;
        }
        if (raw.find('/') != std::string_view::npos || raw.find('\\') != std::string_view::npos) {
            add_error(std::string{path}, std::string{label} + " must not contain path separators");
            return std::nullopt;
        }
        for (const unsigned char ch : raw) {
            if (std::isalnum(ch) == 0 && ch != '_' && ch != '-' && ch != '.') {
                add_error(std::string{path}, std::string{label} + " must use only letters, digits, '_', '-', or '.'");
                return std::nullopt;
            }
        }
        return std::string{raw};
    }

    std::optional<std::string> normalize_run_id(std::string_view raw,
                                                std::string_view path,
                                                std::string_view label) {
        if (raw.empty()) {
            add_error(std::string{path}, std::string{label} + " must not be empty");
            return std::nullopt;
        }
        if (raw.find("..") != std::string_view::npos || raw.find('/') != std::string_view::npos
            || raw.find('\\') != std::string_view::npos) {
            add_error(std::string{path}, std::string{label} + " must not contain path traversal characters");
            return std::nullopt;
        }

        std::string normalized;
        normalized.reserve(raw.size());
        bool has_alnum = false;
        for (const unsigned char ch : raw) {
            if (std::isalnum(ch) != 0) {
                normalized.push_back(static_cast<char>(ch));
                has_alnum = true;
            } else if (ch == '_' || ch == '-') {
                normalized.push_back(static_cast<char>(ch));
            } else {
                normalized.push_back('_');
            }
        }
        if (!has_alnum) {
            add_error(std::string{path}, std::string{label} + " must contain at least one alphanumeric character");
            return std::nullopt;
        }
        return normalized;
    }

    void record_output_name(const std::string &output_name,
                            const std::string &path) {
        const auto [it, inserted] = output_name_paths_.emplace(output_name, path);
        if (!inserted) {
            add_error(path, "duplicate final output name '" + output_name + "' also produced by " + it->second);
        }
    }

    void record_run(const ResolvedRunSpec &run,
                    std::size_t candidate_index) {
        const auto run_id_path = join_path(join_index("resolved_runs", candidate_index), "run_id");
        const auto [run_it, run_inserted] = run_id_paths_.emplace(run.run_id, run_id_path);
        if (!run_inserted) {
            add_error(run_id_path, "duplicate run_id '" + run.run_id + "' also produced by " + run_it->second);
        }

        const auto output_path = run.planned_output_path.generic_string();
        const auto output_path_key = join_path(join_index("resolved_runs", candidate_index), "planned_output_path");
        const auto [path_it, path_inserted] = planned_output_paths_.emplace(output_path, output_path_key);
        if (!path_inserted) {
            add_error(output_path_key,
                      "duplicate planned_output_path '" + output_path + "' also produced by " + path_it->second);
        }
    }

    void expand_experiment(const ExperimentSpec &exp,
                           std::size_t experiment_index) {
        const auto base_path = join_index("experiments", experiment_index);
        const auto id_path = join_path(base_path, "id");

        std::optional<std::string> normalized_id;
        if (exp.id.empty()) {
            add_error(id_path, "experiment id must not be empty");
        } else {
            const auto [it, inserted] = experiment_id_paths_.emplace(exp.id, id_path);
            if (!inserted) {
                add_error(id_path, "duplicate experiment id '" + exp.id + "' also produced by " + it->second);
            }
            normalized_id = normalize_run_id(exp.id, id_path, "experiment id");
        }

        const auto output_name_path = exp.output_name.has_value() ? join_path(base_path, "output_name") : id_path;
        const auto output_name = exp.output_name.has_value()
            ? validate_output_name(*exp.output_name, output_name_path, "output_name")
            : (exp.id.empty()
                   ? std::optional<std::string>{}
                   : validate_output_name(exp.id, output_name_path, "experiment id"));

        const auto repetitions = exp.repetitions.value_or(config_.repetitions);
        if (repetitions < 1) {
            add_error(join_path(base_path, "repetitions"), "experiment repetitions must be at least 1");
            return;
        }
        if (!normalized_id.has_value() || !output_name.has_value()) {
            return;
        }

        record_output_name(*output_name, output_name_path);
        const auto algorithm_budget = budget_or_empty(exp.algorithm_budget);
        const auto optimizer_budget = budget_or_empty(exp.optimizer_budget);

        for (std::size_t repetition_index = 0; repetition_index < repetitions; ++repetition_index) {
            const auto seed = derive_seed(config_, exp, repetition_index);
            if (!seed.has_value()) {
                add_error(join_path(base_path, "seed"),
                          "explicit seed overflows when applying repetition index "
                              + std::to_string(repetition_index));
                break;
            }

            ResolvedRunSpec run;
            run.experiment_id = exp.id;
            run.problem_id = exp.problem;
            run.algorithm_id = exp.algorithm;
            run.optimizer_id = exp.optimizer;
            run.repetition_index = repetition_index;
            run.seed = *seed;
            run.output_name = *output_name;
            run.run_id = *normalized_id + "__rep" + format_repetition_index(repetition_index);
            run.planned_output_path = make_output_path(config_.output_dir, run.output_name, repetition_index);
            run.algorithm_budget = algorithm_budget;
            run.optimizer_budget = optimizer_budget;

            const auto candidate_index = result_.runs.size();
            record_run(run, candidate_index);
            result_.runs.push_back(std::move(run));
        }
    }

    void expand_experiments() {
        for (std::size_t i = 0; i < config_.experiments.size(); ++i) {
            expand_experiment(config_.experiments[i], i);
        }
    }

    const SuiteConfig &config_;
    ExpansionResult result_;
    std::unordered_map<std::string, std::string> experiment_id_paths_;
    std::unordered_map<std::string, std::string> output_name_paths_;
    std::unordered_map<std::string, std::string> run_id_paths_;
    std::unordered_map<std::string, std::string> planned_output_paths_;
};

} // namespace

namespace hpoea::config {

bool ExpansionResult::has_errors() const noexcept {
    for (const auto &diagnostic : diagnostics) {
        if (diagnostic.severity == ExpansionDiagnosticSeverity::Error) {
            return true;
        }
    }
    return false;
}

bool ExpansionResult::ok() const noexcept {
    return !has_errors();
}

ExpansionResult expand_suite_config(const SuiteConfig &config) {
    Expander expander{config};
    return expander.expand();
}

} // namespace hpoea::config
