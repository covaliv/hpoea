#include "dispatch.hpp"

#include "hpoea/config/supported_types.hpp"
#include "hpoea/core/baseline_optimizer.hpp"
#include "hpoea/core/random_search_optimizer.hpp"
#include "hpoea/core/search_space.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#if defined(HPOEA_CONFIG_HAS_PAGMO)
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de1220_algorithm.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/nm_hyper.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sga_algorithm.hpp"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
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

using hpoea::config::benchmark_problem_type_ids;
using hpoea::config::build_has_pagmo;
using hpoea::config::contains;
using hpoea::config::pagmo_algorithm_type_ids;
using hpoea::config::pagmo_optimizer_type_ids;

// cmaes as inner ea has no dispatch case yet
constexpr std::array<std::string_view, 5> cli_algorithm_type_ids{
    "de",
    "sade",
    "pso",
    "sga",
    "de1220"
};

const ProblemSpec *find_problem(const SuiteConfig &config,
                                std::string_view id) {
    for (const auto &problem : config.problems) {
        if (problem.id == id) {
            return &problem;
        }
    }
    return nullptr;
}

hpoea::cli::ComponentDispatch annotate_problem(const ProblemSpec *problem,
                                               std::string_view id) {
    if (!problem) {
        return {std::string{id}, "<missing>", "missing", "unsupported", false};
    }
    if (contains(benchmark_problem_type_ids, problem->type)) {
        return {problem->id, problem->type, "core", "supported", true};
    }
    return {problem->id, problem->type, "custom", "unsupported", false};
}

hpoea::cli::ComponentDispatch annotate_algorithm(const AlgorithmSpec *algorithm,
                                                 std::string_view id) {
    if (!algorithm) {
        return {std::string{id}, "<missing>", "missing", "unsupported", false};
    }
    if (contains(cli_algorithm_type_ids, algorithm->type)) {
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
    if (optimizer->type == "random_search" || optimizer->type == "baseline") {
        return {optimizer->id, optimizer->type, "core", "supported", true};
    }
    if (contains(pagmo_optimizer_type_ids, optimizer->type)) {
        return {optimizer->id,
                optimizer->type,
                build_has_pagmo() ? "pagmo" : "pagmo-unavailable",
                build_has_pagmo() ? "supported" : "requires-pagmo",
                build_has_pagmo()};
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

std::unique_ptr<hpoea::core::IProblem> make_problem(const ProblemSpec &problem,
                                                    std::vector<std::string> &errors) {
    try {
        return hpoea::wrappers::problems::make_benchmark_problem(problem.type, problem.parameters);
    } catch (const std::exception &exception) {
        errors.push_back("problems." + problem.id + ": " + exception.what());
        return nullptr;
    }
}

#if defined(HPOEA_CONFIG_HAS_PAGMO)
std::unique_ptr<hpoea::core::IEvolutionaryAlgorithmFactory> make_pagmo_algorithm_factory(
    std::string_view type) {
    using namespace hpoea::pagmo_wrappers;
    if (type == "de") {
        return std::make_unique<PagmoDifferentialEvolutionFactory>();
    }
    if (type == "sade") {
        return std::make_unique<PagmoSelfAdaptiveDEFactory>();
    }
    if (type == "pso") {
        return std::make_unique<PagmoParticleSwarmOptimizationFactory>();
    }
    if (type == "sga") {
        return std::make_unique<PagmoSgaFactory>();
    }
    if (type == "de1220") {
        return std::make_unique<PagmoDe1220Factory>();
    }
    return nullptr;
}
#endif

std::unique_ptr<hpoea::core::IEvolutionaryAlgorithmFactory> make_algorithm_factory(
    const AlgorithmSpec &algorithm,
    std::vector<std::string> &errors) {
    if (!contains(cli_algorithm_type_ids, algorithm.type)) {
        add_unsupported_error(errors, "algorithm", algorithm.type);
        return nullptr;
    }

#if defined(HPOEA_CONFIG_HAS_PAGMO)
    return make_pagmo_algorithm_factory(algorithm.type);
#else
    errors.push_back("algorithm type requires a Pagmo-enabled build: " + algorithm.type);
    return nullptr;
#endif
}

std::shared_ptr<hpoea::core::SearchSpace> make_algorithm_search_space(
    const AlgorithmSpec &algorithm,
    std::vector<std::string> &errors) {
    if (algorithm.search_parameters.empty()) {
        return nullptr;
    }

    auto search = std::make_shared<hpoea::core::SearchSpace>();
    for (const auto &[name, spec] : algorithm.search_parameters) {
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

std::unique_ptr<hpoea::core::IHyperparameterOptimizer> make_optimizer(
    const OptimizerSpec &optimizer,
    const AlgorithmSpec &algorithm,
    const hpoea::core::IEvolutionaryAlgorithmFactory *algorithm_factory,
    std::vector<std::string> &errors) {
    // make_optimizer is the only place that reports an unsupported optimizer
    auto build_search = [&algorithm, &algorithm_factory, &errors]()
        -> std::shared_ptr<hpoea::core::SearchSpace> {
        if (!algorithm_factory) {
            return nullptr;
        }
        auto search = make_algorithm_search_space(algorithm, errors);
        if (search) {
            search->validate(algorithm_factory->parameter_space());
        }
        return search;
    };

    if (optimizer.type == "random_search") {
        auto random_search = std::make_unique<hpoea::core::RandomSearchOptimizer>();
        if (auto search = build_search()) {
            random_search->set_search_space(std::move(search));
        }
        return random_search;
    }

    if (optimizer.type == "baseline") {
        if (algorithm.fixed_parameters.empty()) {
            return std::make_unique<hpoea::core::BaselineOptimizer>();
        }
        return std::make_unique<hpoea::core::BaselineOptimizer>(algorithm.fixed_parameters);
    }

    if (!contains(pagmo_optimizer_type_ids, optimizer.type)) {
        add_unsupported_error(errors, "optimizer", optimizer.type);
        return nullptr;
    }

#if defined(HPOEA_CONFIG_HAS_PAGMO)
    std::unique_ptr<hpoea::core::HyperOptimizerBase> hyper;
    if (optimizer.type == "cmaes") {
        hyper = std::make_unique<hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer>();
    } else if (optimizer.type == "pso") {
        hyper = std::make_unique<hpoea::pagmo_wrappers::PagmoPsoHyperOptimizer>();
    } else if (optimizer.type == "simulated_annealing") {
        hyper = std::make_unique<hpoea::pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer>();
    } else if (optimizer.type == "nelder_mead") {
        hyper = std::make_unique<hpoea::pagmo_wrappers::PagmoNelderMeadHyperOptimizer>();
    }
    if (!hyper) {
        add_unsupported_error(errors, "optimizer", optimizer.type);
        return nullptr;
    }
    if (auto search = build_search()) {
        hyper->set_search_space(std::move(search));
    }
    return hyper;
#else
    errors.push_back("optimizer type requires a Pagmo-enabled build: " + optimizer.type);
    return nullptr;
#endif
}

} // namespace

namespace hpoea::cli {

// same as the runtime search-space filtering
// fixed and excluded parameters are not tuned
std::optional<std::size_t> tuned_algorithm_dimension(const AlgorithmSpec &algorithm) {
#if defined(HPOEA_CONFIG_HAS_PAGMO)
    const auto factory = make_pagmo_algorithm_factory(algorithm.type);
    if (!factory) {
        return std::nullopt;
    }
    std::size_t dimension = 0;
    for (const auto &descriptor : factory->parameter_space().descriptors()) {
        if (algorithm.fixed_parameters.contains(descriptor.name)) {
            continue;
        }
        const auto it = algorithm.search_parameters.find(descriptor.name);
        if (it != algorithm.search_parameters.end() &&
            it->second.mode == SearchParameterMode::Exclude) {
            continue;
        }
        dimension += 1;
    }
    return dimension;
#else
    (void)algorithm;
    return std::nullopt;
#endif
}

const config::AlgorithmSpec *find_algorithm(const config::SuiteConfig &config,
                                            std::string_view id) noexcept {
    for (const auto &algorithm : config.algorithms) {
        if (algorithm.id == id) {
            return &algorithm;
        }
    }
    return nullptr;
}

const config::OptimizerSpec *find_optimizer(const config::SuiteConfig &config,
                                            std::string_view id) noexcept {
    for (const auto &optimizer : config.optimizers) {
        if (optimizer.id == id) {
            return &optimizer;
        }
    }
    return nullptr;
}

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
        // make_optimizer tolerates a null factory
        // so optimizer error still gets reported
        result.objects.optimizer = make_optimizer(
            *optimizer, *algorithm, result.objects.algorithm_factory.get(), result.errors);
    } catch (const std::exception &exception) {
        result.errors.push_back(exception.what());
    }

    return result;
}

} // namespace hpoea::cli
