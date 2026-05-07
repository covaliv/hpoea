#include "dispatch.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::OptimizerSpec;
using hpoea::config::ProblemSpec;
using hpoea::config::ResolvedRunSpec;
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

} // namespace hpoea::cli
