#include "hpoea/config/config_validator.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::BudgetConfig;
using hpoea::config::ExperimentSpec;
using hpoea::config::OptimizerSpec;
using hpoea::config::ProblemSpec;
using hpoea::config::SearchParameterMode;
using hpoea::config::SearchParameterSpec;
using hpoea::config::SuiteConfig;
using hpoea::config::ValidationDiagnostic;
using hpoea::config::ValidationDiagnosticSeverity;
using hpoea::config::ValidationResult;

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
        validate_experiments();
        validate_algorithms();
        return result_;
    }

private:
    void add_error(std::string path,
                   std::string message) {
        result_.diagnostics.push_back(ValidationDiagnostic{ValidationDiagnosticSeverity::Error,
                                                           std::move(path),
                                                           std::move(message)});
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
            if (!problems_by_id_.emplace(problem.id, &problem).second) {
                add_error(join_path("problems", problem.id), "duplicate problem id '" + problem.id + "'");
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
            if (!algorithms_by_id_.emplace(algorithm.id, &algorithm).second) {
                add_error(join_path("algorithms", algorithm.id), "duplicate algorithm id '" + algorithm.id + "'");
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
            if (!optimizers_by_id_.emplace(optimizer.id, &optimizer).second) {
                add_error(join_path("optimizers", optimizer.id), "duplicate optimizer id '" + optimizer.id + "'");
            }
        }
    }

    void validate_algorithms() {
        for (const auto &algorithm : config_.algorithms) {
            const auto base_path = join_path("algorithms", algorithm.id);
            if (algorithm.type.empty()) {
                add_error(join_path(base_path, "type"), "algorithm type must not be empty");
            }
            for (const auto &[name, spec] : algorithm.search_parameters) {
                const auto path = join_path(join_path(base_path, "search"), name);
                if (algorithm.fixed_parameters.contains(name)) {
                    add_error(path, "parameter appears in both fixed and search parameter sets");
                }
                validate_search_parameter(spec, path);
            }
        }
    }

    void validate_experiments() {
        for (std::size_t i = 0; i < config_.experiments.size(); ++i) {
            const auto &experiment = config_.experiments[i];
            const auto base_path = join_index("experiments", i);
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
                add_error(join_path(base_path, "problem"), "unknown problem '" + experiment.problem + "'");
            }
            if (!algorithms_by_id_.contains(experiment.algorithm)) {
                add_error(join_path(base_path, "algorithm"), "unknown algorithm '" + experiment.algorithm + "'");
            }
            if (!optimizers_by_id_.contains(experiment.optimizer)) {
                add_error(join_path(base_path, "optimizer"), "unknown optimizer '" + experiment.optimizer + "'");
            }
        }
    }

    void validate_budget(const BudgetConfig &budget,
                         const std::string &path) {
        if (budget.generations.has_value() && *budget.generations == 0) {
            add_error(join_path(path, "generations"), "budget generations must be greater than zero");
        }
        if (budget.function_evaluations.has_value() && *budget.function_evaluations == 0) {
            add_error(join_path(path, "function_evaluations"), "budget function_evaluations must be greater than zero");
        }
    }

    void validate_search_parameter(const SearchParameterSpec &spec,
                                   const std::string &path) {
        if (spec.mode == SearchParameterMode::Range) {
            if (!spec.continuous_range.has_value() || spec.continuous_range->lower >= spec.continuous_range->upper) {
                add_error(path, "range mode requires min less than max");
            }
        } else if (spec.mode == SearchParameterMode::IntegerRange) {
            if (!spec.integer_range.has_value() || spec.integer_range->lower >= spec.integer_range->upper) {
                add_error(path, "integer_range mode requires min less than max");
            }
        } else if (spec.mode == SearchParameterMode::Choice && spec.choices.empty()) {
            add_error(join_path(path, "values"), "choice mode requires at least one value");
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
