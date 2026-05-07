#include "dispatch.hpp"

#include "hpoea/core/search_space.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#if defined(HPOEA_CONFIG_HAS_PAGMO)
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <variant>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::ConfigValue;
using hpoea::config::OptimizerSpec;
using hpoea::config::ProblemParameterSet;
using hpoea::config::ProblemSpec;
using hpoea::config::ResolvedRunSpec;
using hpoea::config::SearchParameterMode;
using hpoea::config::SearchParameterSpec;
using hpoea::config::SuiteConfig;

constexpr bool build_has_pagmo() noexcept {
#if defined(HPOEA_CONFIG_HAS_PAGMO)
    return true;
#else
    return false;
#endif
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

const ProblemSpec *find_problem(const SuiteConfig &config,
                                std::string_view id) {
    for (const auto &problem : config.problems) {
        if (problem.id == id) {
            return &problem;
        }
    }
    return nullptr;
}

const AlgorithmSpec *find_algorithm(const SuiteConfig &config,
                                    std::string_view id) {
    for (const auto &algorithm : config.algorithms) {
        if (algorithm.id == id) {
            return &algorithm;
        }
    }
    return nullptr;
}

const OptimizerSpec *find_optimizer(const SuiteConfig &config,
                                    std::string_view id) {
    for (const auto &optimizer : config.optimizers) {
        if (optimizer.id == id) {
            return &optimizer;
        }
    }
    return nullptr;
}

hpoea::cli::ComponentDispatch annotate_problem(const ProblemSpec *problem,
                                               std::string_view id) {
    if (!problem) {
        return {std::string{id}, "<missing>", "missing", "unsupported", false};
    }
    if (problem->type == "sphere") {
        return {problem->id, problem->type, "core", "supported", true};
    }
    return {problem->id, problem->type, "custom", "unsupported", false};
}

hpoea::cli::ComponentDispatch annotate_algorithm(const AlgorithmSpec *algorithm,
                                                 std::string_view id) {
    if (!algorithm) {
        return {std::string{id}, "<missing>", "missing", "unsupported", false};
    }
    if (algorithm->type == "de") {
        return {algorithm->id,
                algorithm->type,
                build_has_pagmo() ? "pagmo" : "pagmo-unavailable",
                build_has_pagmo() ? "supported" : "requires-pagmo",
                build_has_pagmo()};
    }
    if (contains(pagmo_algorithm_type_ids, algorithm->type)) {
        return {algorithm->id,
                algorithm->type,
                build_has_pagmo() ? "pagmo" : "pagmo-unavailable",
                "unsupported",
                false};
    }
    return {algorithm->id, algorithm->type, "custom", "unsupported", false};
}

hpoea::cli::ComponentDispatch annotate_optimizer(const OptimizerSpec *optimizer,
                                                 std::string_view id) {
    if (!optimizer) {
        return {std::string{id}, "<missing>", "missing", "unsupported", false};
    }
    if (optimizer->type == "cmaes") {
        return {optimizer->id,
                optimizer->type,
                build_has_pagmo() ? "pagmo" : "pagmo-unavailable",
                build_has_pagmo() ? "supported" : "requires-pagmo",
                build_has_pagmo()};
    }
    if (contains(pagmo_optimizer_type_ids, optimizer->type)) {
        return {optimizer->id,
                optimizer->type,
                build_has_pagmo() ? "pagmo" : "pagmo-unavailable",
                "unsupported",
                false};
    }
    return {optimizer->id, optimizer->type, "custom", "unsupported", false};
}

void add_missing_error(std::vector<std::string> &errors,
                       std::string_view kind,
                       std::string_view id) {
    errors.push_back(std::string{"missing "} + std::string{kind} + " id: " + std::string{id});
}

void add_unsupported_error(std::vector<std::string> &errors,
                           std::string_view kind,
                           std::string_view type) {
    errors.push_back(std::string{"unsupported dispatch for "}
        + std::string{kind} + " type: " + std::string{type});
}

const ConfigValue *find_parameter(const ProblemParameterSet &parameters,
                                  std::string_view name) {
    for (const auto &[key, value] : parameters) {
        if (key == name) {
            return &value;
        }
    }
    return nullptr;
}

std::optional<std::int64_t> read_integer_parameter(const ProblemParameterSet &parameters,
                                                   std::string_view name,
                                                   std::string_view path,
                                                   std::vector<std::string> &errors) {
    const auto *value = find_parameter(parameters, name);
    if (!value) {
        errors.push_back(std::string{path} + " is required");
        return std::nullopt;
    }
    if (const auto *integer = std::get_if<std::int64_t>(value)) {
        return *integer;
    }
    errors.push_back(std::string{path} + " must be an integer");
    return std::nullopt;
}

std::optional<double> read_optional_number_parameter(const ProblemParameterSet &parameters,
                                                     std::string_view name,
                                                     std::string_view path,
                                                     std::vector<std::string> &errors) {
    const auto *value = find_parameter(parameters, name);
    if (!value) {
        return std::nullopt;
    }
    if (const auto *floating = std::get_if<double>(value)) {
        return *floating;
    }
    if (const auto *integer = std::get_if<std::int64_t>(value)) {
        return static_cast<double>(*integer);
    }
    errors.push_back(std::string{path} + " must be numeric");
    return std::nullopt;
}

std::unique_ptr<hpoea::core::IProblem> make_problem(const ProblemSpec &problem,
                                                    std::vector<std::string> &errors) {
    if (problem.type != "sphere") {
        add_unsupported_error(errors, "problem", problem.type);
        return nullptr;
    }

    const auto dimension = read_integer_parameter(
        problem.parameters, "dimension", "problems." + problem.id + ".dimension", errors);
    const auto lower_bound = read_optional_number_parameter(
        problem.parameters, "lower_bound", "problems." + problem.id + ".lower_bound", errors).value_or(-5.0);
    const auto upper_bound = read_optional_number_parameter(
        problem.parameters, "upper_bound", "problems." + problem.id + ".upper_bound", errors).value_or(5.0);

    if (!dimension.has_value()) {
        return nullptr;
    }
    if (*dimension < 1) {
        errors.push_back("problems." + problem.id + ".dimension must be at least 1");
        return nullptr;
    }
    const auto max_size = static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
    if (static_cast<std::uint64_t>(*dimension) > max_size) {
        errors.push_back("problems." + problem.id + ".dimension is too large");
        return nullptr;
    }

    return std::make_unique<hpoea::wrappers::problems::SphereProblem>(
        static_cast<std::size_t>(*dimension),
        lower_bound,
        upper_bound);
}

std::unique_ptr<hpoea::core::IEvolutionaryAlgorithmFactory> make_algorithm_factory(
    const AlgorithmSpec &algorithm,
    std::vector<std::string> &errors) {
    if (algorithm.type != "de") {
        add_unsupported_error(errors, "algorithm", algorithm.type);
        return nullptr;
    }

#if defined(HPOEA_CONFIG_HAS_PAGMO)
    return std::make_unique<hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory>();
#else
    errors.push_back("algorithm type requires a Pagmo-enabled build: de");
    return nullptr;
#endif
}

#if defined(HPOEA_CONFIG_HAS_PAGMO)
std::shared_ptr<hpoea::core::SearchSpace> make_algorithm_search_space(
    const AlgorithmSpec &algorithm,
    std::vector<std::string> &errors) {
    if (algorithm.search_parameters.empty()) {
        return nullptr;
    }

    auto search = std::make_shared<hpoea::core::SearchSpace>();
    for (const auto &[name, spec] : algorithm.search_parameters) {
        // search-space entries constrain what the hyper-optimizer may tune
        switch (spec.mode) {
        case SearchParameterMode::Range:
            if (!spec.continuous_range.has_value()) {
                errors.push_back("algorithms." + algorithm.id + ".search." + name
                    + " missing range bounds");
                continue;
            }
            search->optimize(name, *spec.continuous_range);
            break;
        case SearchParameterMode::IntegerRange:
            if (!spec.integer_range.has_value()) {
                errors.push_back("algorithms." + algorithm.id + ".search." + name
                    + " missing integer bounds");
                continue;
            }
            search->optimize(name, *spec.integer_range);
            break;
        case SearchParameterMode::Choice:
            search->optimize_choices(name, spec.choices);
            break;
        case SearchParameterMode::Exclude:
            search->exclude(name);
            break;
        }
    }

    return search;
}
#endif

std::unique_ptr<hpoea::core::IHyperparameterOptimizer> make_optimizer(
    const OptimizerSpec &optimizer,
    const AlgorithmSpec &algorithm,
    const hpoea::core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    std::vector<std::string> &errors) {
    if (optimizer.type != "cmaes") {
        add_unsupported_error(errors, "optimizer", optimizer.type);
        return nullptr;
    }

#if defined(HPOEA_CONFIG_HAS_PAGMO)
    auto cmaes = std::make_unique<hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer>();
    auto search = make_algorithm_search_space(algorithm, errors);
    if (search) {
        search->validate(algorithm_factory.parameter_space());
        cmaes->set_search_space(std::move(search));
    }
    return cmaes;
#else
    (void)algorithm;
    (void)algorithm_factory;
    errors.push_back("optimizer type requires a Pagmo-enabled build: cmaes");
    return nullptr;
#endif
}

} // namespace

namespace hpoea::cli {

RunDispatch annotate_run_dispatch(const SuiteConfig &config,
                                  const ResolvedRunSpec &run) {
    RunDispatch annotation;
    annotation.problem = annotate_problem(find_problem(config, run.problem_id), run.problem_id);
    annotation.algorithm = annotate_algorithm(find_algorithm(config, run.algorithm_id), run.algorithm_id);
    annotation.optimizer = annotate_optimizer(find_optimizer(config, run.optimizer_id), run.optimizer_id);
    annotation.runnable = annotation.problem.runnable
        && annotation.algorithm.runnable
        && annotation.optimizer.runnable;
    return annotation;
}

DispatchResult make_dispatch_objects(const SuiteConfig &config,
                                      const ResolvedRunSpec &run) {
    DispatchResult result;
    const auto *problem = find_problem(config, run.problem_id);
    const auto *algorithm = find_algorithm(config, run.algorithm_id);
    const auto *optimizer = find_optimizer(config, run.optimizer_id);

    if (!problem) {
        add_missing_error(result.errors, "problem", run.problem_id);
    }
    if (!algorithm) {
        add_missing_error(result.errors, "algorithm", run.algorithm_id);
    }
    if (!optimizer) {
        add_missing_error(result.errors, "optimizer", run.optimizer_id);
    }
    if (!result.errors.empty()) {
        return result;
    }

    try {
        result.objects.problem = make_problem(*problem, result.errors);
        result.objects.algorithm_factory = make_algorithm_factory(*algorithm, result.errors);
        if (result.objects.algorithm_factory) {
            result.objects.optimizer = make_optimizer(
                *optimizer, *algorithm, *result.objects.algorithm_factory, result.errors);
        } else if (optimizer->type != "cmaes") {
            add_unsupported_error(result.errors, "optimizer", optimizer->type);
        } else if (!build_has_pagmo()) {
            result.errors.push_back("optimizer type requires a Pagmo-enabled build: cmaes");
        }
    } catch (const std::exception &exception) {
        result.errors.push_back(exception.what());
    }

    return result;
}

} // namespace hpoea::cli
