#include "hpoea/core/experiment.hpp"

#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/core/types.hpp"

#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>
#include <pagmo/types.hpp>
#include <random>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>
#include <stdexcept>

namespace hpoea::core {

ExperimentResult SequentialExperimentManager::run_experiment(const ExperimentConfig &config,
                                                             IHyperparameterOptimizer &optimizer,
                                                             const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                             const IProblem &problem,
                                                             ILogger &logger) {
    if (config.trials_per_optimizer == 0) {
        throw std::invalid_argument("trials_per_optimizer must be greater than zero");
    }

    ExperimentResult result;
    result.experiment_id = config.experiment_id;

    ParameterSet optimizer_parameters;
    if (config.optimizer_parameters.has_value()) {
        optimizer_parameters = optimizer.parameter_space().apply_defaults(*config.optimizer_parameters);
    } else {
        optimizer_parameters = optimizer.parameter_space().apply_defaults({});
    }

    optimizer.configure(optimizer_parameters);

    std::random_device device;
    std::mt19937 rng(device());

    for (std::size_t trial = 0; trial < config.trials_per_optimizer; ++trial) {
        const unsigned long optimizer_seed = static_cast<unsigned long>(rng());

        auto optimization_result = optimizer.optimize(algorithm_factory, problem, config.optimizer_budget, optimizer_seed);
        optimization_result.seed = optimizer_seed;
        optimization_result.effective_optimizer_parameters = optimizer_parameters;

        for (const auto &trial_record : optimization_result.trials) {
            RunRecord log_record;
            log_record.experiment_id = config.experiment_id;
            log_record.problem_id = problem.metadata().id;
            log_record.evolutionary_algorithm = algorithm_factory.identity();
            log_record.hyper_optimizer = optimizer.identity();
            log_record.algorithm_parameters = trial_record.parameters;
            log_record.optimizer_parameters = optimizer_parameters;
            log_record.status = trial_record.optimization_result.status;
            log_record.objective_value = trial_record.optimization_result.best_fitness;
            log_record.budget_usage = trial_record.optimization_result.budget_usage;
            log_record.algorithm_seed = trial_record.optimization_result.seed;
            log_record.optimizer_seed = optimizer_seed;
            log_record.message = trial_record.optimization_result.message;

            logger.log(log_record);
        }

        result.optimizer_results.push_back(std::move(optimization_result));
    }

    logger.flush();

    return result;
}

ParallelExperimentManager::ParallelExperimentManager(std::size_t num_threads)
    : num_threads_(num_threads > 0 ? num_threads : std::thread::hardware_concurrency()) {
    if (num_threads_ == 0) {
        num_threads_ = 1;
    }
}

ExperimentResult ParallelExperimentManager::run_experiment(const ExperimentConfig &config,
                                                          IHyperparameterOptimizer &optimizer,
                                                          const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                          const IProblem &problem,
                                                          ILogger &logger) {
    if (config.trials_per_optimizer == 0) {
        throw std::invalid_argument("trials_per_optimizer must be greater than zero");
    }
    if (config.islands == 0) {
        throw std::invalid_argument("islands must be greater than zero");
    }

    ExperimentResult result;
    result.experiment_id = config.experiment_id;

    ParameterSet optimizer_parameters;
    if (config.optimizer_parameters.has_value()) {
        optimizer_parameters = optimizer.parameter_space().apply_defaults(*config.optimizer_parameters);
    } else {
        optimizer_parameters = optimizer.parameter_space().apply_defaults({});
    }

    optimizer.configure(optimizer_parameters);

    std::random_device device;
    std::mt19937 rng(device());
    std::vector<unsigned long> seeds;
    seeds.reserve(config.trials_per_optimizer);
    for (std::size_t i = 0; i < config.trials_per_optimizer; ++i) {
        seeds.push_back(static_cast<unsigned long>(rng()));
    }

    std::vector<HyperparameterOptimizationResult> optimization_results(config.trials_per_optimizer);
    std::mutex logger_mutex;

    const std::size_t num_islands = std::min(config.islands, config.trials_per_optimizer);
    const std::size_t trials_per_island = (config.trials_per_optimizer + num_islands - 1) / num_islands;

    std::vector<std::thread> workers;
    workers.reserve(num_islands);

    for (std::size_t island_idx = 0; island_idx < num_islands; ++island_idx) {
        workers.emplace_back([&, island_idx]() {
            const std::size_t start_trial = island_idx * trials_per_island;
            const std::size_t end_trial = std::min(start_trial + trials_per_island, config.trials_per_optimizer);

            for (std::size_t trial = start_trial; trial < end_trial; ++trial) {
                const unsigned long optimizer_seed = seeds[trial];

                auto optimization_result = optimizer.optimize(algorithm_factory, problem, config.optimizer_budget, optimizer_seed);
                optimization_result.seed = optimizer_seed;
                optimization_result.effective_optimizer_parameters = optimizer_parameters;

                optimization_results[trial] = std::move(optimization_result);

                {
                    std::scoped_lock lock(logger_mutex);
                    for (const auto &trial_record : optimization_results[trial].trials) {
                        RunRecord log_record;
                        log_record.experiment_id = config.experiment_id;
                        log_record.problem_id = problem.metadata().id;
                        log_record.evolutionary_algorithm = algorithm_factory.identity();
                        log_record.hyper_optimizer = optimizer.identity();
                        log_record.algorithm_parameters = trial_record.parameters;
                        log_record.optimizer_parameters = optimizer_parameters;
                        log_record.status = trial_record.optimization_result.status;
                        log_record.objective_value = trial_record.optimization_result.best_fitness;
                        log_record.budget_usage = trial_record.optimization_result.budget_usage;
                        log_record.algorithm_seed = trial_record.optimization_result.seed;
                        log_record.optimizer_seed = optimizer_seed;
                        log_record.message = trial_record.optimization_result.message;

                        logger.log(log_record);
                    }
                }
            }
        });
    }

    for (auto &worker : workers) {
        worker.join();
    }

    result.optimizer_results = std::move(optimization_results);
    logger.flush();

    return result;
}

} // namespace hpoea::core
