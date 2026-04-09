#include "test_harness.hpp"
#include "test_fixtures.hpp"

#include "hpoea/core/experiment.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <memory>
#include <string>

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
    }

    [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override { return identity_; }
    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
        return parameter_space_;
    }
    [[nodiscard]] hpoea::core::HyperparameterOptimizerPtr clone() const override {
        return std::make_unique<DummyOptimizer>(*this);
    }
    void configure(const hpoea::core::ParameterSet &parameters) override {
        configured_ = parameter_space_.apply_defaults(parameters);
    }

    [[nodiscard]] hpoea::core::HyperparameterOptimizationResult optimize(
        const hpoea::core::IEvolutionaryAlgorithmFactory &factory,
        const hpoea::core::IProblem &problem,
        const hpoea::core::Budget &optimizer_budget,
        const hpoea::core::Budget &algorithm_budget,
        unsigned long seed) override {
        auto algo = factory.create();
        hpoea::core::ParameterSet algo_params;
        algo_params.emplace("population_size", std::int64_t{20});
        algo_params.emplace("generations", std::int64_t{2});
        algo->configure(algo_params);

        auto result = algo->run(problem, algorithm_budget, seed);

        hpoea::core::HyperparameterOptimizationResult out;
        out.status = result.status;
        out.best_objective = result.best_fitness;
        out.trials.push_back({algo_params, result});
        out.optimizer_usage.objective_calls = 1;
        out.optimizer_usage.iterations = 0;
        out.effective_optimizer_parameters = configured_;
        out.message = "dummy optimize";
        return out;
    }

private:
    hpoea::core::AlgorithmIdentity identity_{};
    hpoea::core::ParameterSpace parameter_space_{};
    hpoea::core::ParameterSet configured_{};
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
        config.experiment_id = "exp_baseline_ok";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;
        config.algorithm_baseline_parameters = hpoea::core::ParameterSet{{"population_size", std::int64_t{30}}};

        auto optimize_fn = [](const hpoea::core::IEvolutionaryAlgorithmFactory &factory,
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
            out.optimizer_usage.objective_calls = 1;
            out.optimizer_usage.iterations = 0;
            out.message = "stub optimize";
            return out;
        };

        hpoea::tests_v2::StubHyperOptimizer stub_optimizer(optimize_fn);
        hpoea::core::SequentialExperimentManager manager;
        auto result = manager.run_experiment(config, stub_optimizer, factory, problem, logger);
        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == 1u,
                       "experiment returns one optimizer result with baseline parameters");
        HPOEA_V2_CHECK(runner, !logger.records.empty(), "logger records trials for baseline run");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_parallel_invalid";
        config.trials_per_optimizer = 1;
        config.islands = 0;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;

        bool threw = false;
        try {
            hpoea::core::ParallelExperimentManager manager(2);
            (void)manager.run_experiment(config, optimizer, factory, problem, logger);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "parallel manager rejects islands=0");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_parallel_ok";
        config.trials_per_optimizer = 2;
        config.islands = 2;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;

        hpoea::core::ParallelExperimentManager manager(2);
        auto result = manager.run_experiment(config, optimizer, factory, problem, logger);
        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == config.trials_per_optimizer,
                       "parallel manager returns correct number of results");
        bool statuses_ok = true;
        for (const auto &opt_result : result.optimizer_results) {
            if (opt_result.status != hpoea::core::RunStatus::Success &&
                opt_result.status != hpoea::core::RunStatus::BudgetExceeded) {
                statuses_ok = false;
                break;
            }
        }
        HPOEA_V2_CHECK(runner, statuses_ok, "parallel manager results have valid status");
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
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_empty_baseline";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;
        config.algorithm_baseline_parameters = hpoea::core::ParameterSet{};

        hpoea::core::SequentialExperimentManager manager;
        bool threw = false;
        try {
            (void)manager.run_experiment(config, optimizer, factory, problem, logger);
        } catch (const std::exception &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "empty baseline parameters rejected");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_unknown_baseline";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;
        config.algorithm_baseline_parameters = hpoea::core::ParameterSet{{"unknown_param", 1.0}};

        hpoea::core::SequentialExperimentManager manager;
        bool threw = false;
        try {
            (void)manager.run_experiment(config, optimizer, factory, problem, logger);
        } catch (const std::exception &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "unknown baseline parameter rejected");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_all_fixed_baseline";
        config.trials_per_optimizer = 1;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;

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
        config.algorithm_baseline_parameters = baseline;

        hpoea::core::SequentialExperimentManager manager;
        bool threw = false;
        try {
            (void)manager.run_experiment(config, optimizer, factory, problem, logger);
        } catch (const std::exception &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "baseline fixing all parameters rejected");
    }


    {
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "exp_parallel_1worker";
        config.trials_per_optimizer = 2;
        config.islands = 2;
        config.algorithm_budget.generations = 2u;
        config.optimizer_budget.generations = 1u;

        hpoea::core::ParallelExperimentManager manager(1);
        auto result = manager.run_experiment(config, optimizer, factory, problem, logger);
        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == config.trials_per_optimizer,
                       "parallel manager with 1 worker returns correct number of results");
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

    return runner.summarize("experiment_manager_tests");
}
