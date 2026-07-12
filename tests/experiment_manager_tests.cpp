#include "test_harness.hpp"
#include "test_fixtures.hpp"

#include "hpoea/core/experiment.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

class DummyOptimizer final : public hpoea::core::IHyperparameterOptimizer {
public:
    DummyOptimizer() {
        identity_ = {"DummyOpt", "tests", "1.0"};
        hpoea::core::ParameterDescriptor desc;
        desc.name = "generations";
        desc.type = hpoea::core::ParameterType::Integer;
        desc.integer_range = hpoea::core::IntegerRange{1, 10};
        desc.default_value = std::int64_t{2};
        parameter_space_.add_descriptor(desc);
        configured_ = parameter_space_.apply_defaults({});
    }

    [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override { return identity_; }
    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
        return parameter_space_;
    }
    [[nodiscard]] hpoea::core::HyperparameterOptimizerPtr clone() const override {
        ++(*clone_count_);
        return std::make_unique<DummyOptimizer>(*this);
    }
    void configure(const hpoea::core::ParameterSet &parameters) override {
        configured_ = parameter_space_.apply_defaults(parameters);
    }
    [[nodiscard]] const hpoea::core::ParameterSet &configured_parameters() const noexcept override {
        return configured_;
    }

    [[nodiscard]] hpoea::core::HyperparameterOptimizationResult optimize(
        const hpoea::core::IEvolutionaryAlgorithmFactory &factory,
        const hpoea::core::IProblem &problem,
        const hpoea::core::Budget &optimizer_budget,
        const hpoea::core::Budget &algorithm_budget,
        unsigned long seed) override {
        {
            std::scoped_lock lock(*observed_seed_mutex_);
            observed_optimizer_seeds_->push_back(seed);
        }

        auto algo = factory.create();
        hpoea::core::ParameterSet algo_params;
        algo_params.emplace("population_size", std::int64_t{20});
        algo_params.emplace("generations", std::int64_t{2});
        algo->configure(algo_params);

        auto result = algo->run(problem, algorithm_budget, seed);

        hpoea::core::HyperparameterOptimizationResult out;
        out.status = result.status;
        out.best_objective = result.best_fitness;
        out.best_parameters = algo_params;
        out.trials.push_back({algo_params, result});
        out.optimizer_usage.objective_calls = 1;
        out.optimizer_usage.iterations = 0;
        out.effective_optimizer_parameters = configured_;
        out.message = "dummy optimize";
        return out;
    }

    [[nodiscard]] std::size_t clone_count() const noexcept { return *clone_count_; }

    [[nodiscard]] std::vector<unsigned long> observed_optimizer_seeds() const {
        std::scoped_lock lock(*observed_seed_mutex_);
        return *observed_optimizer_seeds_;
    }

private:
    hpoea::core::AlgorithmIdentity identity_{};
    hpoea::core::ParameterSpace parameter_space_{};
    hpoea::core::ParameterSet configured_{};
    std::shared_ptr<std::size_t> clone_count_{std::make_shared<std::size_t>(0)};
    std::shared_ptr<std::vector<unsigned long>> observed_optimizer_seeds_{std::make_shared<std::vector<unsigned long>>()};
    std::shared_ptr<std::mutex> observed_seed_mutex_{std::make_shared<std::mutex>()};
};

}

int main() {
    hpoea::tests_v2::TestRunner runner;
    hpoea::wrappers::problems::SphereProblem problem(3);
    hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    DummyOptimizer optimizer;
    hpoea::tests_v2::CapturingLogger logger;


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_invalid";
        config.trials_per_optimizer = 0;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;

        bool threw = false;
        try {
            hpoea::core::SequentialExperimentManager manager;
            (void)manager.run_experiment(config, optimizer, factory, problem, logger);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "sequential manager rejects trials_per_optimizer=0");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_baseline";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;
        config.algorithm_baseline_parameters = hpoea::core::ParameterSet{{"population_size", std::int64_t{30}}};

        hpoea::core::SequentialExperimentManager manager;
        bool threw = false;
        try {
            (void)manager.run_experiment(config, optimizer, factory, problem, logger);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw,
                       "baseline parameters cannot be overridden by optimizer parameters");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_parallel_invalid";
        config.trials_per_optimizer = 1;
        config.max_parallel_trials = 0;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;

        bool threw = false;
        try {
            hpoea::core::ParallelExperimentManager manager(2);
            (void)manager.run_experiment(config, optimizer, factory, problem, logger);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "parallel manager rejects max_parallel_trials=0");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_parallel_ok";
        config.trials_per_optimizer = 2;
        config.max_parallel_trials = 2;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;

        const auto clones_before = optimizer.clone_count();
        const auto observed_before = optimizer.observed_optimizer_seeds().size();

        hpoea::core::ParallelExperimentManager manager(2);
        auto result = manager.run_experiment(config, optimizer, factory, problem, logger);
        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == config.trials_per_optimizer,
                       "parallel manager returns correct number of results");
        HPOEA_V2_CHECK(runner, optimizer.clone_count() >= clones_before + 1u,
                       "parallel manager uses cloned optimizer instances");
        bool statuses_ok = true;
        std::vector<unsigned long> result_seeds;
        for (const auto &opt_result : result.optimizer_results) {
            if (opt_result.status != hpoea::core::RunStatus::Success) {
                statuses_ok = false;
            }
            result_seeds.push_back(opt_result.seed);
        }
        const auto observed_after = optimizer.observed_optimizer_seeds();
        std::vector<unsigned long> observed_new;
        for (std::size_t i = observed_before; i < observed_after.size(); ++i) {
            observed_new.push_back(observed_after[i]);
        }
        std::sort(observed_new.begin(), observed_new.end());
        std::sort(result_seeds.begin(), result_seeds.end());
        HPOEA_V2_CHECK(runner, statuses_ok, "parallel manager dummy results are Success");
        HPOEA_V2_CHECK(runner, observed_after.size() == observed_before + config.trials_per_optimizer,
                       "parallel manager passes one optimizer seed per trial");
        HPOEA_V2_CHECK(runner, observed_new == result_seeds,
                       "parallel manager result seeds match seeds passed to optimizers");
    }


    {
        hpoea::wrappers::problems::SphereProblem sphere(2);
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer cmaes_hyper;
        hpoea::tests_v2::CapturingLogger capture_logger;

        hpoea::core::ExperimentConfig config;
        config.experiment_id = "field_mapping_test";
        config.trials_per_optimizer = 1;
        config.random_seed = 123UL;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 2u;

        hpoea::core::SequentialExperimentManager manager;
        auto result = manager.run_experiment(config, cmaes_hyper, de_factory, sphere, capture_logger);

        HPOEA_V2_CHECK(runner, !capture_logger.records.empty(),
                       "build_run_record: logger captured at least one record");

        const auto &rec = capture_logger.records[0];


        HPOEA_V2_CHECK(runner, rec.experiment_id == "field_mapping_test",
                       "build_run_record: experiment_id matches config");


        HPOEA_V2_CHECK(runner, rec.problem_id == sphere.metadata().id,
                       "build_run_record: problem_id matches problem metadata");


        HPOEA_V2_CHECK(runner, !rec.evolutionary_algorithm.family.empty(),
                       "build_run_record: evolutionary_algorithm.family not empty");


        HPOEA_V2_CHECK(runner, rec.hyper_optimizer.has_value(),
                       "build_run_record: hyper_optimizer identity present");
        HPOEA_V2_CHECK(runner, !rec.hyper_optimizer->family.empty(),
                       "build_run_record: hyper_optimizer.family not empty");


        HPOEA_V2_CHECK(runner, !rec.algorithm_parameters.empty(),
                       "build_run_record: algorithm_parameters not empty");


        HPOEA_V2_CHECK(runner, !rec.optimizer_parameters.empty(),
                       "build_run_record: optimizer_parameters not empty");


        HPOEA_V2_CHECK(runner, rec.status == hpoea::core::RunStatus::Success ||
                                   rec.status == hpoea::core::RunStatus::BudgetExceeded,
                       "build_run_record: status is Success or BudgetExceeded");


        HPOEA_V2_CHECK(runner, std::isfinite(rec.objective_value),
                       "build_run_record: objective_value is finite");


        HPOEA_V2_CHECK(runner, rec.algorithm_usage.function_evaluations > 0u,
                       "build_run_record: algorithm_usage.function_evaluations > 0");
        HPOEA_V2_CHECK(runner, rec.algorithm_usage.wall_time.count() >= 0,
                       "build_run_record: algorithm_usage.wall_time non-negative");


        HPOEA_V2_CHECK(runner, rec.optimizer_seed.has_value(),
                       "build_run_record: optimizer_seed present");
        HPOEA_V2_CHECK(runner, !result.optimizer_results.empty() &&
                                  rec.optimizer_seed == result.optimizer_results[0].seed,
                       "build_run_record: optimizer_seed matches optimizer result seed");
        HPOEA_V2_CHECK(runner, rec.algorithm_seed != 0UL,
                       "build_run_record: algorithm_seed present");


        HPOEA_V2_CHECK(runner, !rec.message.empty(),
                       "build_run_record: message not empty");


        bool all_ids_match = true;
        for (const auto &r : capture_logger.records) {
            if (r.experiment_id != "field_mapping_test") {
                all_ids_match = false;
                break;
            }
        }
        HPOEA_V2_CHECK(runner, all_ids_match,
                       "build_run_record: all records share experiment_id");
        if (!result.optimizer_results.empty()) {
            HPOEA_V2_CHECK(runner,
                           capture_logger.records.size() == result.optimizer_results[0].trials.size(),
                           "build_run_record: one log record per hyperparameter trial");
            bool record_seeds_match_trials = capture_logger.records.size() == result.optimizer_results[0].trials.size();
            for (std::size_t i = 0; record_seeds_match_trials && i < capture_logger.records.size(); ++i) {
                if (capture_logger.records[i].algorithm_seed !=
                    result.optimizer_results[0].trials[i].optimization_result.seed) {
                    record_seeds_match_trials = false;
                }
            }
            HPOEA_V2_CHECK(runner, record_seeds_match_trials,
                           "build_run_record: algorithm_seed matches each trial seed");
        }
    }


    {
        // baseline with nothing to tune must be rejected
        // that means empty, unknown key, or fully fixed
        auto all_fixed_baseline = [&] {
            hpoea::core::ParameterSet baseline;
            for (const auto &desc : factory.parameter_space().descriptors()) {
                if (desc.default_value.has_value()) {
                    baseline.emplace(desc.name, *desc.default_value);
                } else if (desc.type == hpoea::core::ParameterType::Continuous) {
                    baseline.emplace(desc.name, desc.continuous_range->lower);
                } else if (desc.type == hpoea::core::ParameterType::Integer) {
                    baseline.emplace(desc.name, desc.integer_range->lower);
                } else if (desc.type == hpoea::core::ParameterType::Boolean) {
                    baseline.emplace(desc.name, false);
                } else if (desc.type == hpoea::core::ParameterType::Categorical) {
                    baseline.emplace(desc.name, desc.categorical_choices.front());
                }
            }
            return baseline;
        };

        struct InvalidBaseline {
            const char *experiment_id;
            const char *label;
            hpoea::core::ParameterSet baseline;
        };
        const std::vector<InvalidBaseline> cases = {
            {"exp_empty_baseline", "empty baseline parameters rejected", hpoea::core::ParameterSet{}},
            {"exp_unknown_baseline", "unknown baseline parameter rejected",
             hpoea::core::ParameterSet{{"unknown_param", 1.0}}},
            {"exp_all_fixed_baseline", "baseline fixing all parameters rejected", all_fixed_baseline()},
        };
        for (const auto &tc : cases) {
            hpoea::core::ExperimentConfig config;
            config.experiment_id = tc.experiment_id;
            config.trials_per_optimizer = 1;
            config.algorithm_budget.generations = 2u;
            config.optimizer_budget.generations = 1u;
            config.algorithm_baseline_parameters = tc.baseline;

            hpoea::core::SequentialExperimentManager manager;
            bool threw = false;
            try {
                (void)manager.run_experiment(config, optimizer, factory, problem, logger);
            } catch (const std::exception &) {
                threw = true;
            }
            HPOEA_V2_CHECK(runner, threw, tc.label);
        }
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_parallel_1worker";
        config.trials_per_optimizer = 2;
        config.max_parallel_trials = 2;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;

        const auto clones_before = optimizer.clone_count();
        hpoea::core::ParallelExperimentManager manager(1);
        auto result = manager.run_experiment(config, optimizer, factory, problem, logger);
        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == config.trials_per_optimizer,
                       "parallel manager with 1 worker returns correct number of results");
        HPOEA_V2_CHECK(runner, optimizer.clone_count() > clones_before,
                       "parallel manager with 1 worker uses cloned optimizer");
    }


    {
        hpoea::wrappers::problems::SphereProblem sphere5(5);
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer cmaes_optimizer;
        hpoea::tests_v2::CapturingLogger baseline_logger;

        hpoea::core::ExperimentConfig config;
        config.experiment_id = "baseline_propagation_test";
        config.trials_per_optimizer = 1;
        config.random_seed = 777UL;
        config.algorithm_budget.generations = 10u;
        config.optimizer_budget.generations = 3u;
        config.algorithm_baseline_parameters = hpoea::core::ParameterSet{
            {"population_size", std::int64_t{30}},
            {"scaling_factor", 0.7},
        };

        hpoea::core::SequentialExperimentManager manager;
        auto experiment = manager.run_experiment(config, cmaes_optimizer, de_factory, sphere5, baseline_logger);

        bool has_results = !experiment.optimizer_results.empty() &&
                           !experiment.optimizer_results[0].trials.empty();
        HPOEA_V2_CHECK(runner, has_results, "baseline experiment produces results");

        bool params_propagated = true;
        if (has_results) {
            for (const auto &trial : experiment.optimizer_results[0].trials) {
                const auto it_pop = trial.optimization_result.effective_parameters.find("population_size");
                const auto it_sf = trial.optimization_result.effective_parameters.find("scaling_factor");
                if (it_pop == trial.optimization_result.effective_parameters.end() ||
                    it_sf == trial.optimization_result.effective_parameters.end()) {
                    params_propagated = false;
                    break;
                }
            }
        }
        HPOEA_V2_CHECK(runner, params_propagated,
                       "baseline parameters propagated through effective_parameters");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_actual_seed";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;


        hpoea::core::SequentialExperimentManager manager;
        auto result = manager.run_experiment(config, optimizer, factory, problem, logger);
        HPOEA_V2_CHECK(runner, result.actual_seed != 0,
                       "actual_seed is captured when random_seed is unset");


        hpoea::core::ExperimentConfig config2;
        config2.experiment_id = "exp_actual_seed_replay";
        config2.trials_per_optimizer = 1;
        config2.algorithm_budget.generations = 2u;
        config2.optimizer_budget.generations = 1u;
        config2.random_seed = result.actual_seed;

        hpoea::tests_v2::CapturingLogger logger2;
        auto result2 = manager.run_experiment(config2, optimizer, factory, problem, logger2);
        HPOEA_V2_CHECK(runner, result2.actual_seed == result.actual_seed,
                       "replayed experiment uses the same actual_seed");

        bool seeds_match = !result.optimizer_results.empty() &&
                           !result2.optimizer_results.empty() &&
                           result.optimizer_results[0].seed == result2.optimizer_results[0].seed;
        HPOEA_V2_CHECK(runner, seeds_match,
                       "replayed experiment produces the same optimizer seed");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_actual_seed_explicit";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;
        config.random_seed = 42UL;

        hpoea::core::SequentialExperimentManager manager;
        hpoea::tests_v2::CapturingLogger explicit_logger;
        auto result = manager.run_experiment(config, optimizer, factory, problem, explicit_logger);
        HPOEA_V2_CHECK(runner, result.actual_seed == 42UL,
                       "actual_seed equals random_seed when explicitly set");
    }


    {
        DummyOptimizer seed_optimizer;
        const auto before = seed_optimizer.observed_optimizer_seeds().size();

        hpoea::core::ExperimentConfig config_a;
        config_a.experiment_id = "seed_low32_a";
        config_a.trials_per_optimizer = 1;
        config_a.algorithm_budget.generations = 2u;
        config_a.optimizer_budget.generations = 1u;
        config_a.random_seed = 7UL;

        hpoea::core::ExperimentConfig config_b = config_a;
        config_b.experiment_id = "seed_low32_b";
        config_b.random_seed = 0x100000007UL;

        hpoea::tests_v2::CapturingLogger logger_a;
        hpoea::tests_v2::CapturingLogger logger_b;
        hpoea::core::SequentialExperimentManager manager;
        (void)manager.run_experiment(config_a, seed_optimizer, factory, problem, logger_a);
        (void)manager.run_experiment(config_b, seed_optimizer, factory, problem, logger_b);

        const auto observed = seed_optimizer.observed_optimizer_seeds();
        HPOEA_V2_CHECK(runner, observed.size() == before + 2u,
                       "two low-32-agreeing seeds each produce one optimizer seed");
        HPOEA_V2_CHECK(runner, observed[before] != observed[before + 1],
                       "seeds agreeing in low 32 bits produce different first optimizer seeds");
    }


    {
        auto zero_trial_fn = [](const hpoea::core::IEvolutionaryAlgorithmFactory &,
                                const hpoea::core::IProblem &,
                                const hpoea::core::Budget &,
                                const hpoea::core::Budget &,
                                unsigned long) {
            hpoea::core::HyperparameterOptimizationResult out;
            out.status = hpoea::core::RunStatus::FailedEvaluation;
            out.error_info = hpoea::core::ErrorInfo{"evaluation_failure", "stub", "no trials"};
            out.message = "zero-trial error";
            return out;
        };

        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_zero_trial";
        config.trials_per_optimizer = 3;
        config.algorithm_budget.generations = 2u;

        config.max_parallel_trials = 1;
        hpoea::tests_v2::StubHyperOptimizer seq_stub(zero_trial_fn);
        hpoea::tests_v2::CapturingLogger seq_logger;
        hpoea::core::SequentialExperimentManager seq_manager;
        auto seq_result = seq_manager.run_experiment(config, seq_stub, factory, problem, seq_logger);

        config.max_parallel_trials = 2;
        hpoea::tests_v2::StubHyperOptimizer par_stub(zero_trial_fn);
        hpoea::tests_v2::CapturingLogger par_logger;
        hpoea::core::ParallelExperimentManager par_manager(2);
        auto par_result = par_manager.run_experiment(config, par_stub, factory, problem, par_logger);

        HPOEA_V2_CHECK(runner, seq_result.optimizer_results.size() == 3u,
                       "sequential returns all zero-trial results");
        HPOEA_V2_CHECK(runner, par_result.optimizer_results.size() == 3u,
                       "parallel no longer drops zero-trial results");
        bool statuses_match = seq_result.optimizer_results.size() == par_result.optimizer_results.size();
        for (std::size_t i = 0; statuses_match && i < seq_result.optimizer_results.size(); ++i) {
            statuses_match = seq_result.optimizer_results[i].status == hpoea::core::RunStatus::FailedEvaluation
                          && par_result.optimizer_results[i].status == hpoea::core::RunStatus::FailedEvaluation;
        }
        HPOEA_V2_CHECK(runner, statuses_match,
                       "sequential and parallel zero-trial results share matching statuses");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_uneven_workers";
        config.trials_per_optimizer = 5;
        config.max_parallel_trials = 4;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;
        config.random_seed = 2024UL;

        DummyOptimizer parallel_optimizer;
        hpoea::tests_v2::CapturingLogger parallel_logger;
        hpoea::core::ParallelExperimentManager manager(4);
        auto result = manager.run_experiment(config, parallel_optimizer, factory, problem, parallel_logger);
        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == 5u,
                       "trials 5 workers 4 completes with 5 results");

        hpoea::core::ExperimentConfig seq_config = config;
        seq_config.max_parallel_trials = 1;
        DummyOptimizer sequential_optimizer;
        hpoea::tests_v2::CapturingLogger sequential_logger;
        hpoea::core::SequentialExperimentManager sequential_manager;
        auto seq_result = sequential_manager.run_experiment(seq_config, sequential_optimizer, factory, problem,
                                                            sequential_logger);
        bool order_matches = seq_result.optimizer_results.size() == result.optimizer_results.size();
        for (std::size_t i = 0; order_matches && i < result.optimizer_results.size(); ++i) {
            order_matches = result.optimizer_results[i].seed == seq_result.optimizer_results[i].seed;
        }
        HPOEA_V2_CHECK(runner, order_matches,
                       "parallel uneven-worker results are in the same seed order as sequential");
    }


    {
        // parallel and sequential must match trial for trial
        // each trial's seed is fixed up front
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_parallel_equals_sequential";
        config.trials_per_optimizer = 4;
        config.max_parallel_trials = 2;
        config.algorithm_budget.generations = 3u;
        config.optimizer_budget.generations = 1u;
        config.random_seed = 7331UL;

        DummyOptimizer parallel_optimizer;
        hpoea::tests_v2::CapturingLogger parallel_logger;
        hpoea::core::ParallelExperimentManager parallel_manager(2);
        auto par_result = parallel_manager.run_experiment(config, parallel_optimizer, factory, problem,
                                                          parallel_logger);

        hpoea::core::ExperimentConfig seq_config = config;
        seq_config.max_parallel_trials = 1;
        DummyOptimizer sequential_optimizer;
        hpoea::tests_v2::CapturingLogger sequential_logger;
        hpoea::core::SequentialExperimentManager sequential_manager;
        auto seq_result = sequential_manager.run_experiment(seq_config, sequential_optimizer, factory, problem,
                                                            sequential_logger);

        bool sizes_match = par_result.optimizer_results.size() == seq_result.optimizer_results.size()
                        && par_result.optimizer_results.size() == config.trials_per_optimizer;
        HPOEA_V2_CHECK(runner, sizes_match,
                       "parallel and sequential return the same trial count");

        bool trials_match = sizes_match;
        for (std::size_t i = 0; trials_match && i < par_result.optimizer_results.size(); ++i) {
            const auto &par = par_result.optimizer_results[i];
            const auto &seq = seq_result.optimizer_results[i];
            trials_match = par.seed == seq.seed
                        && par.status == seq.status
                        && par.best_objective == seq.best_objective
                        && par.trials.size() == seq.trials.size();
        }
        HPOEA_V2_CHECK(runner, trials_match,
                       "parallel per-trial seed, status, best_objective, and trial count match sequential");
    }


    {
        hpoea::core::ParameterSet marker_params;
        marker_params.emplace("clamped_marker", std::int64_t{42});
        auto marker_fn = [marker_params](const hpoea::core::IEvolutionaryAlgorithmFactory &factory,
                                         const hpoea::core::IProblem &problem,
                                         const hpoea::core::Budget &,
                                         const hpoea::core::Budget &algorithm_budget,
                                         unsigned long seed) {
            auto algo = factory.create();
            hpoea::core::ParameterSet algo_params;
            algo_params.emplace("generations", std::int64_t{2});
            algo->configure(algo_params);
            auto run_result = algo->run(problem, algorithm_budget, seed);
            hpoea::core::HyperparameterOptimizationResult out;
            out.status = run_result.status;
            out.best_objective = run_result.best_fitness;
            out.trials.push_back({algo_params, run_result});
            out.effective_optimizer_parameters = marker_params;
            out.message = "marker optimize";
            return out;
        };

        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_effective_params";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        config.optimizer_parameters = hpoea::core::ParameterSet{};

        hpoea::tests_v2::StubHyperOptimizer marker_stub(marker_fn);
        hpoea::tests_v2::CapturingLogger marker_logger;
        hpoea::core::SequentialExperimentManager manager;
        auto result = manager.run_experiment(config, marker_stub, factory, problem, marker_logger);
        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == 1u,
                       "effective-params experiment returns one result");
        HPOEA_V2_CHECK(runner, !result.optimizer_results.empty() &&
                                   result.optimizer_results[0].effective_optimizer_parameters.contains("clamped_marker"),
                       "manager preserves effective_optimizer_parameters from the optimizer");
    }


    {
        DummyOptimizer optimizer;
        hpoea::core::ParameterSet direct;
        direct.emplace("generations", std::int64_t{7});
        optimizer.configure(direct);

        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_preconfigured_optimizer";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        // config.optimizer_parameters is unset here
        // manager must keep the direct configuration

        hpoea::tests_v2::CapturingLogger preconfigured_logger;
        hpoea::core::SequentialExperimentManager manager;
        auto result = manager.run_experiment(config, optimizer, factory, problem, preconfigured_logger);

        bool kept = !result.optimizer_results.empty();
        if (kept) {
            const auto &effective = result.optimizer_results[0].effective_optimizer_parameters;
            const auto it = effective.find("generations");
            kept = it != effective.end() && std::get<std::int64_t>(it->second) == 7;
        }
        HPOEA_V2_CHECK(runner, kept,
                       "manager keeps directly configured optimizer parameters when config sets none");

        bool logged = !preconfigured_logger.records.empty();
        if (logged) {
            const auto &params = preconfigured_logger.records.front().optimizer_parameters;
            const auto it = params.find("generations");
            logged = it != params.end() && std::get<std::int64_t>(it->second) == 7;
        }
        HPOEA_V2_CHECK(runner, logged,
                       "run records log the optimizer's actual configured parameters when config sets none");
    }

    return runner.summarize("experiment_manager_tests");
}
