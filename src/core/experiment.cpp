#include "hpoea/core/experiment.hpp"

#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/core/types.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>
#include <stdexcept>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::EvolutionaryAlgorithmPtr;
using hpoea::core::ExperimentConfig;
using hpoea::core::HyperparameterTrialRecord;
using hpoea::core::IEvolutionaryAlgorithm;
using hpoea::core::IEvolutionaryAlgorithmFactory;
using hpoea::core::IProblem;
using hpoea::core::ParameterSet;
using hpoea::core::ParameterValidationError;
using hpoea::core::RunRecord;

ParameterSet resolve_optimizer_parameters(const hpoea::core::IHyperparameterOptimizer &optimizer,
                                          const std::optional<ParameterSet> &overrides) {
    if (overrides) {
        return optimizer.parameter_space().apply_defaults(*overrides);
    }
    return optimizer.parameter_space().apply_defaults({});
}

ParameterSet validate_baseline_parameters(const IEvolutionaryAlgorithmFactory &factory,
                                          const ParameterSet &baseline_parameters) {
    if (baseline_parameters.empty()) {
        throw std::invalid_argument("algorithm_baseline_parameters must not be empty when provided");
    }

    hpoea::core::ParameterSpace subset_space;
    for (const auto &[name, _] : baseline_parameters) {
        if (!factory.parameter_space().contains(name)) {
            throw ParameterValidationError("unknown baseline parameter: " + name);
        }
        subset_space.add_descriptor(factory.parameter_space().descriptor(name));
    }
    return subset_space.apply_defaults(baseline_parameters);
}

hpoea::core::ParameterSpace filter_parameter_space(const hpoea::core::ParameterSpace &source,
                                                   const ParameterSet &fixed_parameters) {
    hpoea::core::ParameterSpace filtered;
    for (const auto &descriptor : source.descriptors()) {
        if (!fixed_parameters.contains(descriptor.name)) {
            filtered.add_descriptor(descriptor);
        }
    }
    return filtered;
}

class BaselineAppliedAlgorithm final : public IEvolutionaryAlgorithm {
public:
    BaselineAppliedAlgorithm(EvolutionaryAlgorithmPtr inner,
                             ParameterSet baseline_parameters,
                             hpoea::core::ParameterSpace exposed_parameter_space)
        : inner_(std::move(inner)),
          baseline_parameters_(std::move(baseline_parameters)),
          exposed_parameter_space_(std::move(exposed_parameter_space)) {
        if (!inner_) {
            throw std::invalid_argument("baseline-applied algorithm requires a non-null inner algorithm");
        }
    }

    [[nodiscard]] const AlgorithmIdentity &identity() const noexcept override {
        return inner_->identity();
    }

    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
        return exposed_parameter_space_;
    }

    void configure(const ParameterSet &parameters) override {
        for (const auto &[name, _] : parameters) {
            if (baseline_parameters_.contains(name)) {
                throw std::invalid_argument("baseline parameter cannot be overridden: " + name);
            }
        }

        ParameterSet merged = parameters;
        for (const auto &[name, value] : baseline_parameters_) {
            merged.insert_or_assign(name, value);
        }

        merged = inner_->parameter_space().apply_defaults(merged);
        inner_->configure(merged);
        configured_parameters_ = std::move(merged);
    }

    [[nodiscard]] hpoea::core::OptimizationResult run(const IProblem &problem,
                                                      const hpoea::core::Budget &budget,
                                                      unsigned long seed) override {
        auto result = inner_->run(problem, budget, seed);
        if (result.effective_parameters.empty()) {
            result.effective_parameters = configured_parameters_;
        }
        return result;
    }

    [[nodiscard]] std::unique_ptr<IEvolutionaryAlgorithm> clone() const override {
        return std::make_unique<BaselineAppliedAlgorithm>(
            inner_->clone(), baseline_parameters_, exposed_parameter_space_);
    }

private:
    EvolutionaryAlgorithmPtr inner_;
    ParameterSet baseline_parameters_;
    ParameterSet configured_parameters_;
    hpoea::core::ParameterSpace exposed_parameter_space_;
};

class BaselineAppliedFactory final : public IEvolutionaryAlgorithmFactory {
public:
    BaselineAppliedFactory(const IEvolutionaryAlgorithmFactory &base_factory,
                           ParameterSet baseline_parameters)
        : base_factory_(base_factory),
          baseline_parameters_(std::move(baseline_parameters)),
          filtered_parameter_space_(filter_parameter_space(base_factory.parameter_space(), baseline_parameters_)) {
        if (filtered_parameter_space_.empty()) {
            throw std::invalid_argument(
                "algorithm_baseline_parameters fixes all parameters; no tunable parameters remain for optimization");
        }
    }

    [[nodiscard]] EvolutionaryAlgorithmPtr create() const override {
        return std::make_unique<BaselineAppliedAlgorithm>(
            base_factory_.create(), baseline_parameters_, filtered_parameter_space_);
    }

    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
        return filtered_parameter_space_;
    }

    [[nodiscard]] const AlgorithmIdentity &identity() const noexcept override {
        return base_factory_.identity();
    }

private:
    const IEvolutionaryAlgorithmFactory &base_factory_;
    ParameterSet baseline_parameters_;
    hpoea::core::ParameterSpace filtered_parameter_space_;
};

const IEvolutionaryAlgorithmFactory &resolve_algorithm_factory(
    const IEvolutionaryAlgorithmFactory &base_factory,
    const std::optional<ParameterSet> &baseline_parameters,
    std::unique_ptr<IEvolutionaryAlgorithmFactory> &owned_factory) {
    if (!baseline_parameters.has_value()) {
        return base_factory;
    }

    auto validated_baseline = validate_baseline_parameters(base_factory, *baseline_parameters);
    owned_factory = std::make_unique<BaselineAppliedFactory>(base_factory, std::move(validated_baseline));
    return *owned_factory;
}

std::pair<std::mt19937, unsigned long> seed_rng(const std::optional<unsigned long> &random_seed) {
    unsigned long actual_seed;
    if (random_seed) {
        actual_seed = *random_seed;
    } else {
        std::random_device device;
        auto lo = static_cast<unsigned long>(device());
        auto hi = static_cast<unsigned long>(device());
        actual_seed = (hi << 32) | lo;
    }
    std::mt19937 rng(actual_seed);
    return {rng, actual_seed};
}

ParameterSet select_logged_parameters(const HyperparameterTrialRecord &trial_record) {
    if (!trial_record.optimization_result.effective_parameters.empty()) {
        return trial_record.optimization_result.effective_parameters;
    }
    return trial_record.parameters;
}

RunRecord build_run_record(const ExperimentConfig &config,
                           const IProblem &problem,
                           const AlgorithmIdentity &algorithm_identity,
                           const AlgorithmIdentity &optimizer_identity,
                           const ParameterSet &optimizer_parameters,
                           unsigned long optimizer_seed,
                           const HyperparameterTrialRecord &trial_record) {
    RunRecord log_record;
    log_record.experiment_id = config.experiment_id;
    log_record.problem_id = problem.metadata().id;
    log_record.evolutionary_algorithm = algorithm_identity;
    log_record.hyper_optimizer = optimizer_identity;
    log_record.algorithm_parameters = select_logged_parameters(trial_record);
    log_record.optimizer_parameters = optimizer_parameters;
    log_record.status = trial_record.optimization_result.status;
    log_record.objective_value = trial_record.optimization_result.best_fitness;
    log_record.requested_budget = trial_record.optimization_result.requested_budget;
    log_record.effective_budget = trial_record.optimization_result.effective_budget;
    log_record.algorithm_usage = trial_record.optimization_result.algorithm_usage;
    log_record.error_info = trial_record.optimization_result.error_info;
    log_record.algorithm_seed = trial_record.optimization_result.seed;
    log_record.optimizer_seed = optimizer_seed;
    log_record.message = trial_record.optimization_result.message;
    return log_record;
}

void run_island_trials(
    std::size_t island_idx,
    std::size_t trials_per_island,
    const ExperimentConfig &config,
    const std::vector<unsigned long> &seeds,
    const IEvolutionaryAlgorithmFactory &active_algorithm_factory,
    const IProblem &problem,
    const ParameterSet &optimizer_parameters,
    std::vector<hpoea::core::HyperparameterOptimizationResult> &optimization_results,
    hpoea::core::IHyperparameterOptimizer &worker_optimizer,
    hpoea::core::ILogger &logger,
    std::mutex &logger_mutex,
    std::atomic<bool> &stop_requested) {
    const std::size_t start_trial = island_idx * trials_per_island;
    const std::size_t end_trial = std::min(start_trial + trials_per_island, config.trials_per_optimizer);

    for (std::size_t trial = start_trial; trial < end_trial; ++trial) {
        if (stop_requested.load(std::memory_order_relaxed)) break;
        const unsigned long optimizer_seed = seeds[trial];

        auto optimization_result = worker_optimizer.optimize(
            active_algorithm_factory,
            problem,
            config.optimizer_budget,
            config.algorithm_budget,
            optimizer_seed);
        optimization_result.seed = optimizer_seed;
        optimization_result.effective_optimizer_parameters = optimizer_parameters;

        optimization_results[trial] = std::move(optimization_result);

        {
            std::scoped_lock lock(logger_mutex);
            for (const auto &trial_record : optimization_results[trial].trials) {
                logger.log(build_run_record(
                    config,
                    problem,
                    active_algorithm_factory.identity(),
                    worker_optimizer.identity(),
                    optimizer_parameters,
                    optimizer_seed,
                    trial_record));
            }
        }
    }
}

} // namespace

namespace hpoea::core {

ExperimentResult SequentialExperimentManager::run_experiment(const ExperimentConfig &config,
                                                             IHyperparameterOptimizer &optimizer,
                                                             const IEvolutionaryAlgorithmFactory &algorithm_factory,
                                                             const IProblem &problem,
                                                             ILogger &logger) {
    if (config.trials_per_optimizer == 0) {
        throw std::invalid_argument("trials_per_optimizer must be greater than zero");
    }
    if (config.islands > 1) {
        throw std::invalid_argument(
            "sequential experiment manager does not support multiple islands; "
            "use ParallelExperimentManager or set islands to 1");
    }

    std::unique_ptr<IEvolutionaryAlgorithmFactory> owned_factory;
    const auto &active_algorithm_factory =
        resolve_algorithm_factory(algorithm_factory, config.algorithm_baseline_parameters, owned_factory);

    ExperimentResult result;
    result.experiment_id = config.experiment_id;

    ParameterSet optimizer_parameters = resolve_optimizer_parameters(optimizer, config.optimizer_parameters);

    optimizer.configure(optimizer_parameters);

    auto [rng, actual_seed] = seed_rng(config.random_seed);
    result.actual_seed = actual_seed;

    for (std::size_t trial = 0; trial < config.trials_per_optimizer; ++trial) {
        const unsigned long optimizer_seed = static_cast<unsigned long>(rng());

        auto optimization_result = optimizer.optimize(
            active_algorithm_factory,
            problem,
            config.optimizer_budget,
            config.algorithm_budget,
            optimizer_seed);
        optimization_result.seed = optimizer_seed;
        optimization_result.effective_optimizer_parameters = optimizer_parameters;

        for (const auto &trial_record : optimization_result.trials) {
            logger.log(build_run_record(
                config,
                problem,
                active_algorithm_factory.identity(),
                optimizer.identity(),
                optimizer_parameters,
                optimizer_seed,
                trial_record));
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

    std::unique_ptr<IEvolutionaryAlgorithmFactory> owned_factory;
    const auto &active_algorithm_factory =
        resolve_algorithm_factory(algorithm_factory, config.algorithm_baseline_parameters, owned_factory);

    ExperimentResult result;
    result.experiment_id = config.experiment_id;

    ParameterSet optimizer_parameters = resolve_optimizer_parameters(optimizer, config.optimizer_parameters);

    optimizer.configure(optimizer_parameters);

    auto [rng, actual_seed] = seed_rng(config.random_seed);
    result.actual_seed = actual_seed;
    std::vector<unsigned long> seeds;
    seeds.reserve(config.trials_per_optimizer);
    for (std::size_t i = 0; i < config.trials_per_optimizer; ++i) {
        seeds.push_back(static_cast<unsigned long>(rng()));
    }

    std::vector<HyperparameterOptimizationResult> optimization_results(config.trials_per_optimizer);
    std::mutex logger_mutex;
    std::mutex worker_error_mutex;
    std::vector<std::pair<std::size_t, std::exception_ptr>> worker_errors;
    std::atomic<bool> stop_requested{false};

    const std::size_t num_islands = std::min({config.islands, config.trials_per_optimizer, num_threads_});
    const std::size_t trials_per_island = (config.trials_per_optimizer + num_islands - 1) / num_islands;

    std::vector<std::thread> workers;
    workers.reserve(num_islands);

    for (std::size_t island_idx = 0; island_idx < num_islands; ++island_idx) {
        auto worker_optimizer = optimizer.clone();
        if (!worker_optimizer) {
            throw std::runtime_error("IHyperparameterOptimizer::clone() returned null");
        }

        workers.emplace_back([&, island_idx, worker_optimizer = std::move(worker_optimizer)]() mutable {
            try {
                run_island_trials(island_idx, trials_per_island, config, seeds,
                                  active_algorithm_factory, problem, optimizer_parameters,
                                  optimization_results, *worker_optimizer, logger,
                                  logger_mutex, stop_requested);
            } catch (...) {
                stop_requested.store(true, std::memory_order_relaxed);
                std::scoped_lock lock(worker_error_mutex);
                worker_errors.emplace_back(island_idx, std::current_exception());
            }
        });
    }

    for (auto &worker : workers) {
        worker.join();
    }

    if (!worker_errors.empty()) {
        std::string combined = std::to_string(worker_errors.size()) + " worker(s) failed in parallel experiment:";
        for (const auto &[island, eptr] : worker_errors) {
            combined += "\n  [island " + std::to_string(island) + "]: ";
            try {
                std::rethrow_exception(eptr);
            } catch (const std::exception &ex) {
                combined += ex.what();
            } catch (...) {
                combined += "unknown exception";
            }
        }
        throw std::runtime_error(combined);
    }

    std::size_t dropped = 0;
    for (auto &opt_result : optimization_results) {
        if (!opt_result.trials.empty()) {
            result.optimizer_results.push_back(std::move(opt_result));
        } else {
            ++dropped;
        }
    }
    if (dropped > 0 && !result.optimizer_results.empty()) {
        auto &last = result.optimizer_results.back();
        last.message += " (" + std::to_string(dropped) + " trial(s) dropped due to early termination)";
    }
    logger.flush();

    return result;
}

} // namespace hpoea::core
