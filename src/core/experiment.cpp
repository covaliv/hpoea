#include "hpoea/core/experiment.hpp"

#include "hpoea/core/error_classification.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/core/seeding.hpp"
#include "hpoea/core/types.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
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

// reconfigure when overrides are passed
// otherwise keep and log the caller's configuration
ParameterSet resolve_optimizer_parameters(hpoea::core::IHyperparameterOptimizer &optimizer,
                                          const std::optional<ParameterSet> &overrides) {
    if (overrides) {
        optimizer.configure(*overrides);
    }
    return optimizer.configured_parameters();
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

std::pair<std::mt19937_64, unsigned long> seed_rng(const std::optional<unsigned long> &random_seed) {
    unsigned long actual_seed;
    if (random_seed) {
        actual_seed = *random_seed;
    } else {
        std::random_device device;
        const auto lo = static_cast<std::uint64_t>(device());
        const auto hi = static_cast<std::uint64_t>(device());
        actual_seed = static_cast<unsigned long>((hi << 32) | lo);
    }
    std::mt19937_64 rng(hpoea::core::splitmix64(static_cast<std::uint64_t>(actual_seed)));
    return {rng, actual_seed};
}

ParameterSet select_logged_parameters(const HyperparameterTrialRecord &trial_record) {
    if (!trial_record.optimization_result.effective_parameters.empty()) {
        return trial_record.optimization_result.effective_parameters;
    }
    return trial_record.parameters;
}

// distinct stream from tuning trial seeds
constexpr std::uint64_t validation_stream_salt = 0x0a11da7e5eed5a17ULL;

bool has_selectable_trial(const hpoea::core::HyperparameterOptimizationResult &result) {
    return std::any_of(result.trials.begin(), result.trials.end(),
                       hpoea::core::is_selectable_trial);
}

void run_validation_repeats(const ExperimentConfig &config,
                            const IEvolutionaryAlgorithmFactory &active_algorithm_factory,
                            const IProblem &problem,
                            unsigned long optimizer_seed,
                            hpoea::core::HyperparameterOptimizationResult &optimization_result) {
    // selectable winner validated even when cell ended budget_exceeded
    if (config.validation_repeats == 0 ||
        optimization_result.best_parameters.empty() ||
        !has_selectable_trial(optimization_result)) {
        return;
    }

    const auto validation_base =
        hpoea::core::splitmix64(static_cast<std::uint64_t>(optimizer_seed) ^ validation_stream_salt);
    optimization_result.validation_runs.reserve(config.validation_repeats);
    for (std::size_t i = 0; i < config.validation_repeats; ++i) {
        const auto validation_seed =
            static_cast<unsigned long>(hpoea::core::derive_stream_seed(validation_base, i));
        hpoea::core::OptimizationResult validation_run;
        try {
            auto algorithm = active_algorithm_factory.create();
            algorithm->configure(optimization_result.best_parameters);
            validation_run = algorithm->run(problem, config.algorithm_budget, validation_seed);
        } catch (const std::exception &ex) {
            const auto classified = hpoea::core::classify_exception(ex);
            validation_run.status = classified.status;
            validation_run.error_info = classified.error_info;
            validation_run.message = ex.what();
        }
        validation_run.seed = validation_seed;
        optimization_result.validation_runs.push_back(std::move(validation_run));
    }
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

// trial i uses seeds[i] and slot i
// deterministic under any thread timing
void run_trials(
    const ExperimentConfig &config,
    const std::vector<unsigned long> &seeds,
    const IEvolutionaryAlgorithmFactory &active_algorithm_factory,
    const IProblem &problem,
    const ParameterSet &optimizer_parameters,
    std::vector<hpoea::core::HyperparameterOptimizationResult> &optimization_results,
    hpoea::core::IHyperparameterOptimizer &worker_optimizer,
    hpoea::core::ILogger &logger,
    std::mutex &logger_mutex,
    std::atomic<std::size_t> &next_trial_index,
    std::atomic<bool> &stop_requested) {
    for (;;) {
        if (stop_requested.load(std::memory_order_relaxed)) break;
        const std::size_t trial = next_trial_index.fetch_add(1, std::memory_order_relaxed);
        if (trial >= config.trials_per_optimizer) break;
        const unsigned long optimizer_seed = seeds[trial];

        auto optimization_result = worker_optimizer.optimize(
            active_algorithm_factory,
            problem,
            config.optimizer_budget,
            config.algorithm_budget,
            optimizer_seed);
        optimization_result.seed = optimizer_seed;
        if (optimization_result.effective_optimizer_parameters.empty()) {
            optimization_result.effective_optimizer_parameters = optimizer_parameters;
        }

        run_validation_repeats(config, active_algorithm_factory, problem, optimizer_seed, optimization_result);

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
            for (const auto &validation_run : optimization_results[trial].validation_runs) {
                auto log_record = build_run_record(
                    config,
                    problem,
                    active_algorithm_factory.identity(),
                    worker_optimizer.identity(),
                    optimizer_parameters,
                    optimizer_seed,
                    {optimization_results[trial].best_parameters, validation_run});
                log_record.phase = hpoea::core::RunPhase::Validation;
                logger.log(log_record);
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
    if (config.max_parallel_trials > 1) {
        throw std::invalid_argument(
            "sequential experiment manager runs trials in the calling thread; "
            "set max_parallel_trials to 1 or use ParallelExperimentManager");
    }

    std::unique_ptr<IEvolutionaryAlgorithmFactory> owned_factory;
    const auto &active_algorithm_factory =
        resolve_algorithm_factory(algorithm_factory, config.algorithm_baseline_parameters, owned_factory);

    ExperimentResult result;
    result.experiment_id = config.experiment_id;

    ParameterSet optimizer_parameters = resolve_optimizer_parameters(optimizer, config.optimizer_parameters);

    auto [rng, actual_seed] = seed_rng(config.random_seed);
    result.actual_seed = actual_seed;

    std::vector<unsigned long> seeds;
    seeds.reserve(config.trials_per_optimizer);
    for (std::size_t i = 0; i < config.trials_per_optimizer; ++i) {
        seeds.push_back(static_cast<unsigned long>(rng()));
    }

    std::vector<HyperparameterOptimizationResult> optimization_results(config.trials_per_optimizer);
    std::mutex logger_mutex;
    std::atomic<std::size_t> next_trial_index{0};
    std::atomic<bool> stop_requested{false};

    run_trials(config, seeds, active_algorithm_factory, problem, optimizer_parameters,
               optimization_results, optimizer, logger, logger_mutex, next_trial_index, stop_requested);

    result.optimizer_results = std::move(optimization_results);

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
    if (config.max_parallel_trials == 0) {
        throw std::invalid_argument("max_parallel_trials must be greater than zero");
    }

    std::unique_ptr<IEvolutionaryAlgorithmFactory> owned_factory;
    const auto &active_algorithm_factory =
        resolve_algorithm_factory(algorithm_factory, config.algorithm_baseline_parameters, owned_factory);

    ExperimentResult result;
    result.experiment_id = config.experiment_id;

    ParameterSet optimizer_parameters = resolve_optimizer_parameters(optimizer, config.optimizer_parameters);

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
    std::atomic<std::size_t> next_trial_index{0};
    std::atomic<bool> stop_requested{false};

    const std::size_t num_workers = std::min({config.max_parallel_trials, config.trials_per_optimizer, num_threads_});

    // clone every worker optimizer before spawning any thread
    // a clone throw mid-spawn would leave joinable threads behind
    // that calls std::terminate
    std::vector<HyperparameterOptimizerPtr> worker_optimizers;
    worker_optimizers.reserve(num_workers);
    for (std::size_t worker_idx = 0; worker_idx < num_workers; ++worker_idx) {
        auto worker_optimizer = optimizer.clone();
        if (!worker_optimizer) {
            throw std::runtime_error("IHyperparameterOptimizer::clone() returned null");
        }
        worker_optimizers.push_back(std::move(worker_optimizer));
    }

    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for (std::size_t worker_idx = 0; worker_idx < num_workers; ++worker_idx) {
        workers.emplace_back([&, worker_idx]() {
            try {
                run_trials(config, seeds, active_algorithm_factory, problem, optimizer_parameters,
                           optimization_results, *worker_optimizers[worker_idx], logger, logger_mutex,
                           next_trial_index, stop_requested);
            } catch (...) {
                stop_requested.store(true, std::memory_order_relaxed);
                std::scoped_lock lock(worker_error_mutex);
                worker_errors.emplace_back(worker_idx, std::current_exception());
            }
        });
    }

    for (auto &worker : workers) {
        worker.join();
    }

    if (!worker_errors.empty()) {
        std::string combined = std::to_string(worker_errors.size()) + " worker(s) failed in parallel experiment:";
        for (const auto &[worker, eptr] : worker_errors) {
            combined += "\n  [worker " + std::to_string(worker) + "]: ";
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

    result.optimizer_results = std::move(optimization_results);
    logger.flush();

    return result;
}

} // namespace hpoea::core
