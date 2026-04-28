#include "hpoea/config/config_validator.hpp"

#include "hpoea/config/suite_expander.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::BudgetConfig;
using hpoea::config::ExperimentSpec;
using hpoea::config::ExpansionDiagnosticSeverity;
using hpoea::config::OptimizerSpec;
using hpoea::config::ProblemSpec;
using hpoea::config::SearchParameterMode;
using hpoea::config::SearchParameterSpec;
using hpoea::config::SuiteConfig;
using hpoea::config::ValidationDiagnostic;
using hpoea::config::ValidationDiagnosticSeverity;
using hpoea::config::ValidationResult;

template <std::size_t Size>
bool contains(const std::array<std::string_view, Size> &ids,
              std::string_view type_id) noexcept {
    for (const auto id : ids) {
        if (id == type_id) {
            return true;
        }
    }
    return false;
}

constexpr std::array<std::string_view, 6> pagmo_algorithm_type_ids{
    "de",
    "pso",
    "sade",
    "sga",
    "de1220",
    "cmaes"
};

constexpr std::array<std::string_view, 4> pagmo_optimizer_type_ids{
    "cmaes",
    "pso",
    "simulated_annealing",
    "nelder_mead"
};

constexpr bool build_has_pagmo() noexcept {
#if defined(HPOEA_CONFIG_HAS_PAGMO)
    return true;
#else
    return false;
#endif
}

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

class Validator {
public:
    explicit Validator(const SuiteConfig &config)
        : config_(config) {}

    ValidationResult validate() {
        validate_suite();
        index_problems();
        index_algorithms();
        index_optimizers();
        validate_problems();
        validate_algorithms();
        validate_optimizers();
        validate_experiments();
        validate_expansion_plan();
        return result_;
    }

private:
    void add_diagnostic(ValidationDiagnosticSeverity severity,
                        std::string path,
                        std::string message) {
        result_.diagnostics.push_back(ValidationDiagnostic{severity, std::move(path), std::move(message)});
    }

    void add_error(std::string path,
                   std::string message) {
        add_diagnostic(ValidationDiagnosticSeverity::Error, std::move(path), std::move(message));
    }

    void validate_suite() {
        if (config_.schema_version != 1) {
            add_error("schema_version", "unsupported config schema version: " + std::to_string(config_.schema_version));
        }
        if (config_.name.empty()) {
            add_error("suite.name", "suite name must not be empty");
        }
        if (config_.output_dir.empty()) {
            add_error("suite.output_dir", "suite output_dir must not be empty");
        }
        if (config_.repetitions < 1) {
            add_error("suite.repetitions", "suite repetitions must be at least 1");
        }
        if (config_.experiments.empty()) {
            add_error("experiments", "at least one explicit experiment is required");
        }
    }

    void index_problems() {
        for (std::size_t i = 0; i < config_.problems.size(); ++i) {
            const auto &problem = config_.problems[i];
            if (problem.id.empty()) {
                add_error(join_path(join_index("problems", i), "id"), "problem id must not be empty");
                continue;
            }
            const auto [it, inserted] = problems_by_id_.emplace(problem.id, &problem);
            if (!inserted) {
                add_error(join_path("problems", problem.id),
                          "duplicate problem id '" + problem.id + "' also produced by " + it->first);
            }
        }
    }

    void index_algorithms() {
        for (std::size_t i = 0; i < config_.algorithms.size(); ++i) {
            const auto &algorithm = config_.algorithms[i];
            if (algorithm.id.empty()) {
                add_error(join_path(join_index("algorithms", i), "id"), "algorithm id must not be empty");
                continue;
            }
            const auto [it, inserted] = algorithms_by_id_.emplace(algorithm.id, &algorithm);
            if (!inserted) {
                add_error(join_path("algorithms", algorithm.id),
                          "duplicate algorithm id '" + algorithm.id + "' also produced by " + it->first);
            }
        }
    }

    void index_optimizers() {
        for (std::size_t i = 0; i < config_.optimizers.size(); ++i) {
            const auto &optimizer = config_.optimizers[i];
            if (optimizer.id.empty()) {
                add_error(join_path(join_index("optimizers", i), "id"), "optimizer id must not be empty");
                continue;
            }
            const auto [it, inserted] = optimizers_by_id_.emplace(optimizer.id, &optimizer);
            if (!inserted) {
                add_error(join_path("optimizers", optimizer.id),
                          "duplicate optimizer id '" + optimizer.id + "' also produced by " + it->first);
            }
        }
    }

    void validate_problems() {
        for (const auto &problem : config_.problems) {
            const auto base_path = join_path("problems", problem.id);
            validate_problem_type(problem.type, join_path(base_path, "type"));
        }
    }

    void validate_problem_type(const std::string &type,
                               const std::string &path) {
        if (type.empty()) {
            add_error(path, "problem type must not be empty");
        }
    }

    void validate_algorithms() {
        for (const auto &algorithm : config_.algorithms) {
            const auto base_path = join_path("algorithms", algorithm.id);
            validate_algorithm_type(algorithm.type, join_path(base_path, "type"));
            for (const auto &[name, spec] : algorithm.search_parameters) {
                const auto path = join_path(join_path(base_path, "search"), name);
                if (algorithm.fixed_parameters.contains(name)) {
                    add_error(path, "parameter appears in both fixed and search parameter sets");
                }
                validate_search_parameter(spec, path);
            }
        }
    }

    void validate_algorithm_type(const std::string &type,
                                 const std::string &path) {
        if (type.empty()) {
            add_error(path, "algorithm type must not be empty");
            return;
        }
        if (contains(pagmo_algorithm_type_ids, type) && !build_has_pagmo()) {
            add_error(path, "algorithm type requires a Pagmo-enabled build: " + type);
        }
    }

    void validate_optimizers() {
        for (const auto &optimizer : config_.optimizers) {
            const auto base_path = join_path("optimizers", optimizer.id);
            validate_optimizer_type(optimizer.type, join_path(base_path, "type"));
        }
    }

    void validate_optimizer_type(const std::string &type,
                                 const std::string &path) {
        if (type.empty()) {
            add_error(path, "optimizer type must not be empty");
            return;
        }
        if (contains(pagmo_optimizer_type_ids, type) && !build_has_pagmo()) {
            add_error(path, "optimizer type requires a Pagmo-enabled build: " + type);
        }
    }

    void validate_experiments() {
        for (std::size_t i = 0; i < config_.experiments.size(); ++i) {
            const auto &experiment = config_.experiments[i];
            const auto base_path = join_index("experiments", i);
            validate_experiment(experiment, base_path);
        }
    }

    void validate_experiment(const ExperimentSpec &experiment,
                             const std::string &base_path) {
        if (experiment.id.empty()) {
            add_error(join_path(base_path, "id"), "experiment id must not be empty");
        }
        if (experiment.repetitions.has_value() && *experiment.repetitions < 1) {
            add_error(join_path(base_path, "repetitions"), "experiment repetitions must be at least 1");
        }
        if (experiment.algorithm_budget.has_value()) {
            validate_budget(*experiment.algorithm_budget, join_path(base_path, "algorithm_budget"));
        }
        if (experiment.optimizer_budget.has_value()) {
            validate_budget(*experiment.optimizer_budget, join_path(base_path, "optimizer_budget"));
        }
        if (!problems_by_id_.contains(experiment.problem)) {
            add_error(join_path(base_path, "problem"),
                      "experiment '" + experiment.id + "': unknown problem '" + experiment.problem + "'");
        }
        if (!algorithms_by_id_.contains(experiment.algorithm)) {
            add_error(join_path(base_path, "algorithm"),
                      "experiment '" + experiment.id + "': unknown algorithm '" + experiment.algorithm + "'");
        }
        if (!optimizers_by_id_.contains(experiment.optimizer)) {
            add_error(join_path(base_path, "optimizer"),
                      "experiment '" + experiment.id + "': unknown optimizer '" + experiment.optimizer + "'");
        }
    }

    void validate_budget(const BudgetConfig &budget,
                         const std::string &path) {
        if (budget.generations.has_value() && *budget.generations == 0) {
            add_error(join_path(path, "generations"), "budget generations must be greater than zero");
        }
        if (budget.function_evaluations.has_value() && *budget.function_evaluations == 0) {
            add_error(join_path(path, "function_evaluations"),
                      "budget function_evaluations must be greater than zero");
        }
    }

    bool search_has_bounds(const SearchParameterSpec &spec) const noexcept {
        return spec.min_present || spec.max_present
            || spec.continuous_range.has_value() || spec.integer_range.has_value();
    }

    bool search_has_partial_bounds(const SearchParameterSpec &spec) const noexcept {
        return spec.min_present != spec.max_present;
    }

    void reject_non_choice_values(const SearchParameterSpec &spec,
                                  const std::string &path) {
        if (!spec.choices.empty()) {
            add_error(join_path(path, "values"), "values are only supported for choice search parameters");
        }
    }

    void validate_search_parameter(const SearchParameterSpec &spec,
                                   const std::string &path) {
        switch (spec.mode) {
            case SearchParameterMode::Range:
                validate_range_search_parameter(spec, path);
                return;
            case SearchParameterMode::IntegerRange:
                validate_integer_range_search_parameter(spec, path);
                return;
            case SearchParameterMode::Choice:
                validate_choice_search_parameter(spec, path);
                return;
            case SearchParameterMode::Exclude:
                validate_exclude_search_parameter(spec, path);
                return;
        }
        add_error(path, "unsupported search parameter mode");
    }

    void validate_range_search_parameter(const SearchParameterSpec &spec,
                                         const std::string &path) {
        reject_non_choice_values(spec, path);
        if (search_has_partial_bounds(spec)) {
            add_error(path, "range mode requires both min and max");
            return;
        }
        if (!spec.continuous_range.has_value()) {
            add_error(path, "range mode requires both min and max");
            return;
        }
        if (spec.integer_range.has_value()) {
            add_error(path, "range mode must not define integer bounds");
        }
        if (spec.continuous_range->lower >= spec.continuous_range->upper) {
            add_error(path, "range min must be less than max");
        }
    }

    void validate_integer_range_search_parameter(const SearchParameterSpec &spec,
                                                 const std::string &path) {
        reject_non_choice_values(spec, path);
        if (search_has_partial_bounds(spec)) {
            add_error(path, "integer_range mode requires both min and max");
            return;
        }
        if (!spec.integer_range.has_value()) {
            add_error(path, "integer_range mode requires both min and max");
            return;
        }
        if (spec.continuous_range.has_value()) {
            add_error(path, "integer_range mode must not define floating-point bounds");
        }
        if (spec.integer_range->lower >= spec.integer_range->upper) {
            add_error(path, "integer_range min must be less than max");
        }
    }

    void validate_choice_search_parameter(const SearchParameterSpec &spec,
                                          const std::string &path) {
        if (spec.choices.empty()) {
            add_error(join_path(path, "values"), "choice mode requires at least one value");
        }
        if (search_has_bounds(spec)) {
            add_error(path, "choice mode must not define min or max bounds");
        }
    }

    void validate_exclude_search_parameter(const SearchParameterSpec &spec,
                                           const std::string &path) {
        if (search_has_bounds(spec)) {
            add_error(path, "exclude mode must not define min or max bounds");
        }
        reject_non_choice_values(spec, path);
    }

    void validate_expansion_plan() {
        const auto expansion = hpoea::config::expand_suite_config(config_);
        for (const auto &diag : expansion.diagnostics) {
            const auto severity = diag.severity == ExpansionDiagnosticSeverity::Error
                ? ValidationDiagnosticSeverity::Error
                : ValidationDiagnosticSeverity::Warning;
            add_diagnostic(severity, diag.path, diag.message);
        }
    }

    const SuiteConfig &config_;
    ValidationResult result_;
    std::unordered_map<std::string, const ProblemSpec *> problems_by_id_;
    std::unordered_map<std::string, const AlgorithmSpec *> algorithms_by_id_;
    std::unordered_map<std::string, const OptimizerSpec *> optimizers_by_id_;
};

} // namespace

namespace hpoea::config {

bool ValidationResult::has_errors() const noexcept {
    for (const auto &diagnostic : diagnostics) {
        if (diagnostic.severity == ValidationDiagnosticSeverity::Error) {
            return true;
        }
    }
    return false;
}

bool ValidationResult::ok() const noexcept {
    return !has_errors();
}

ValidationResult validate_suite_config(const SuiteConfig &config) {
    Validator validator{config};
    return validator.validate();
}

} // namespace hpoea::config
