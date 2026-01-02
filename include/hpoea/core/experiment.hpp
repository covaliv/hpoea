#pragma once

#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/core/types.hpp"

#include <filesystem>
#include <optional>
#include <thread>
#include <vector>

namespace hpoea::core {

struct ExperimentConfig {
    std::string experiment_id;
    std::size_t islands{1};
    std::size_t trials_per_optimizer{1};
    Budget algorithm_budget{};
    Budget optimizer_budget{};
    std::optional<ParameterSet> optimizer_parameters;
    std::optional<ParameterSet> algorithm_baseline_parameters;
    std::filesystem::path log_file_path;
    std::optional<unsigned long> random_seed;
};

struct ExperimentResult {
    std::string experiment_id;
    std::vector<HyperparameterOptimizationResult> optimizer_results;
};

class IExperimentManager {
public:
    virtual ~IExperimentManager() = default;

    [[nodiscard]] virtual ExperimentResult run_experiment(const ExperimentConfig &config,
                                                          IHyperparameterOptimizer &optimizer,
                                                          const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                          const IProblem &problem,
                                                          ILogger &logger) = 0;
};

class SequentialExperimentManager final : public IExperimentManager {
public:
    [[nodiscard]] ExperimentResult run_experiment(const ExperimentConfig &config,
                                                  IHyperparameterOptimizer &optimizer,
                                                  const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                  const IProblem &problem,
                                                  ILogger &logger) override;
};

class ParallelExperimentManager final : public IExperimentManager {
public:
    explicit ParallelExperimentManager(std::size_t num_threads = std::thread::hardware_concurrency());

    [[nodiscard]] ExperimentResult run_experiment(const ExperimentConfig &config,
                                                  IHyperparameterOptimizer &optimizer,
                                                  const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                  const IProblem &problem,
                                                  ILogger &logger) override;

private:
    std::size_t num_threads_;
};

} // namespace hpoea::core

