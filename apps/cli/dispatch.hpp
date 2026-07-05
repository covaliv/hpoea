#pragma once

#include "hpoea/config/config_types.hpp"
#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/problem.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hpoea::cli {

struct ComponentDispatch {
    std::string id;
    std::string type;
    std::string backend;
    std::string dispatch;
    bool runnable{false};
};

struct RunDispatch {
    ComponentDispatch problem;
    ComponentDispatch algorithm;
    ComponentDispatch optimizer;
    bool runnable{false};
};

struct DispatchObjects {
    std::unique_ptr<core::IProblem> problem;
    std::unique_ptr<core::IEvolutionaryAlgorithmFactory> algorithm_factory;
    std::unique_ptr<core::IHyperparameterOptimizer> optimizer;
};

struct DispatchResult {
    DispatchObjects objects;
    std::vector<std::string> errors;

    [[nodiscard]] bool ok() const noexcept {
        return errors.empty()
            && objects.problem != nullptr
            && objects.algorithm_factory != nullptr
            && objects.optimizer != nullptr;
    }
};

[[nodiscard]] RunDispatch annotate_run_dispatch(const config::SuiteConfig &config,
                                                const config::ResolvedRunSpec &run);

[[nodiscard]] DispatchResult make_dispatch_objects(const config::SuiteConfig &config,
                                                   const config::ResolvedRunSpec &run);

[[nodiscard]] const config::AlgorithmSpec *find_algorithm(const config::SuiteConfig &config,
                                                          std::string_view id) noexcept;

[[nodiscard]] const config::OptimizerSpec *find_optimizer(const config::SuiteConfig &config,
                                                          std::string_view id) noexcept;

[[nodiscard]] std::optional<std::size_t> tuned_algorithm_dimension(
    const config::AlgorithmSpec &algorithm);

} // namespace hpoea::cli
