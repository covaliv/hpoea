#include "test_harness.hpp"
#include "test_fixtures.hpp"
#include "test_utils.hpp"

#include "hpoea/core/baseline_optimizer.hpp"
#include "hpoea/core/experiment.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

int main() {
    hpoea::tests_v2::TestRunner runner;
    hpoea::wrappers::problems::SphereProblem problem(3);
    hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;


    {
        hpoea::core::BaselineOptimizer baseline;

        HPOEA_V2_CHECK(runner, baseline.identity().family == "Baseline",
                       "identity family is Baseline");
        HPOEA_V2_CHECK(runner, baseline.identity().implementation == "default_parameters",
                       "identity implementation is default_parameters");
        HPOEA_V2_CHECK(runner, baseline.parameter_space().empty(),
                       "baseline has no tunable parameters");

        hpoea::core::Budget algo_budget;
        algo_budget.generations = 10u;

        auto result = baseline.optimize(factory, problem, {}, algo_budget, 42);

        HPOEA_V2_CHECK(runner, result.trials.size() == 1u,
                       "default baseline produces exactly 1 trial");
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "baseline run returns Success");
        HPOEA_V2_CHECK(runner, std::isfinite(result.best_objective),
                       "baseline produces a finite best objective");
        HPOEA_V2_CHECK(runner, !result.best_parameters.empty(),
                       "baseline populates best_parameters");
        HPOEA_V2_CHECK(runner, result.optimizer_usage.objective_calls == 1,
                       "baseline reports exactly one objective call");
        HPOEA_V2_CHECK(runner, result.message == "baseline run with default parameters",
                       "message indicates default parameters");


        const auto defaults = factory.parameter_space().apply_defaults({});
        HPOEA_V2_CHECK(runner,
                       hpoea::tests_v2::parameter_set_equals(result.trials[0].parameters, defaults),
                       "trial parameters match EA defaults");
    }


    {
        hpoea::core::ParameterSet custom;
        custom.emplace("population_size", std::int64_t{30});
        custom.emplace("crossover_rate", 0.9);
        custom.emplace("scaling_factor", 0.5);

        hpoea::core::BaselineOptimizer baseline(custom);

        HPOEA_V2_CHECK(runner, baseline.identity().implementation == "fixed_parameters",
                       "fixed baseline identity says fixed_parameters");

        hpoea::core::Budget algo_budget;
        algo_budget.generations = 5u;

        auto result = baseline.optimize(factory, problem, {}, algo_budget, 123);

        HPOEA_V2_CHECK(runner, result.trials.size() == 1u,
                       "fixed baseline produces exactly 1 trial");


        const auto &trial_params = result.trials[0].parameters;
        HPOEA_V2_CHECK(runner,
                       hpoea::tests_v2::parameter_value_equals(
                           trial_params.at("population_size"), hpoea::core::ParameterValue{std::int64_t{30}}),
                       "fixed population_size=30 is preserved");
        HPOEA_V2_CHECK(runner,
                       hpoea::tests_v2::parameter_value_equals(
                           trial_params.at("crossover_rate"), hpoea::core::ParameterValue{0.9}),
                       "fixed crossover_rate=0.9 is preserved");
        HPOEA_V2_CHECK(runner, result.message == "baseline run with fixed parameters",
                       "message indicates fixed parameters");
    }


    {
        hpoea::core::ParameterSet custom;
        custom.emplace("population_size", std::int64_t{25});
        hpoea::core::BaselineOptimizer baseline(custom);

        auto cloned = baseline.clone();
        HPOEA_V2_CHECK(runner, cloned != nullptr, "clone returns non-null");
        HPOEA_V2_CHECK(runner, cloned->identity().family == "Baseline",
                       "cloned identity is Baseline");

        hpoea::core::Budget algo_budget;
        algo_budget.generations = 3u;

        auto result = cloned->optimize(factory, problem, {}, algo_budget, 999);
        HPOEA_V2_CHECK(runner, result.trials.size() == 1u,
                       "cloned baseline produces 1 trial");
        HPOEA_V2_CHECK(runner,
                       hpoea::tests_v2::parameter_value_equals(
                           result.trials[0].parameters.at("population_size"),
                           hpoea::core::ParameterValue{std::int64_t{25}}),
                       "cloned baseline preserves fixed parameters");
    }


    {
        hpoea::core::BaselineOptimizer baseline;
        hpoea::tests_v2::CapturingLogger logger;

        hpoea::core::ExperimentConfig config;
        config.experiment_id = "baseline_comparison";
        config.trials_per_optimizer = 3;
        config.algorithm_budget.generations = 5u;
        config.random_seed = 42;

        hpoea::core::SequentialExperimentManager manager;
        auto result = manager.run_experiment(config, baseline, factory, problem, logger);

        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == 3u,
                       "experiment produces 3 optimizer results (one per trial)");
        HPOEA_V2_CHECK(runner, logger.records.size() == 3u,
                       "logger captures 3 run records");


        bool all_baseline = true;
        for (const auto &record : logger.records) {
            if (!record.hyper_optimizer.has_value() ||
                record.hyper_optimizer->family != "Baseline") {
                all_baseline = false;
                break;
            }
        }
        HPOEA_V2_CHECK(runner, all_baseline,
                       "all log records identify Baseline as the hyper optimizer");
    }


    {
        hpoea::core::BaselineOptimizer b1, b2;
        hpoea::core::Budget algo_budget;
        algo_budget.generations = 10u;

        auto r1 = b1.optimize(factory, problem, {}, algo_budget, 777);
        auto r2 = b2.optimize(factory, problem, {}, algo_budget, 777);

        HPOEA_V2_CHECK(runner,
                       hpoea::tests_v2::nearly_equal(r1.best_objective, r2.best_objective),
                       "same seed produces identical best_objective");
    }


    {
        hpoea::tests_v2::ThrowingProblem throwing_problem;

        hpoea::core::BaselineOptimizer baseline;
        hpoea::core::Budget algo_budget;
        algo_budget.generations = 3u;

        auto result = baseline.optimize(factory, throwing_problem, {}, algo_budget, 42);
        HPOEA_V2_CHECK(runner, result.status != hpoea::core::RunStatus::Success,
                       "baseline with throwing problem returns non-Success status");
        HPOEA_V2_CHECK(runner, !result.message.empty(),
                       "baseline with throwing problem has error message");
    }




    {
        struct ThrowingFactory final : hpoea::core::IEvolutionaryAlgorithmFactory {
            [[nodiscard]] hpoea::core::EvolutionaryAlgorithmPtr create() const override {
                throw std::runtime_error("factory create() failed intentionally");
            }
            [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
                return space_;
            }
            [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override {
                return id_;
            }
            hpoea::core::ParameterSpace space_;
            hpoea::core::AlgorithmIdentity id_{"ThrowingFactory", "tests", "1.0"};
        };

        ThrowingFactory throwing_factory;
        hpoea::core::BaselineOptimizer baseline;
        hpoea::core::Budget algo_budget;
        algo_budget.generations = 3u;

        auto result = baseline.optimize(throwing_factory, problem, {}, algo_budget, 42);

        HPOEA_V2_CHECK(runner, result.status != hpoea::core::RunStatus::Success,
                       "failed baseline should not report Success");
        HPOEA_V2_CHECK(runner, !result.trials.empty(),
                       "failed baseline must still produce a trial record for logging");
        if (!result.trials.empty()) {
            HPOEA_V2_CHECK(runner,
                           result.trials[0].optimization_result.status != hpoea::core::RunStatus::Success,
                           "trial record should carry the error status");
        }
    }


    {
        hpoea::core::BaselineOptimizer baseline;
        hpoea::core::Budget algo_budget;
        algo_budget.generations = 5u;

        auto result = baseline.optimize(factory, problem, {}, algo_budget, 42);
        HPOEA_V2_CHECK(runner, result.optimizer_usage.wall_time.count() >= 0,
                       "baseline wall_time is non-negative");
    }


    {
        hpoea::core::BaselineOptimizer baseline;
        hpoea::core::Budget algo_budget;
        algo_budget.generations = 3u;

        auto result = baseline.optimize(factory, problem, {}, algo_budget, 12345);
        HPOEA_V2_CHECK(runner, result.seed == 12345UL,
                       "baseline records seed in result");
    }


    {
        hpoea::core::BaselineOptimizer baseline;
        hpoea::core::Budget algo_budget;
        algo_budget.generations = 5u;


        hpoea::core::Budget optimizer_budget;
        optimizer_budget.function_evaluations = 0u;

        auto result = baseline.optimize(factory, problem, optimizer_budget, algo_budget, 42);

        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::BudgetExceeded,
                       "baseline respects optimizer_budget.function_evaluations");
        HPOEA_V2_CHECK(runner, result.message.find("function-evaluations") != std::string::npos
                            || result.message.find("budget") != std::string::npos,
                       "baseline budget-exceeded message is informative");
    }


    {
        hpoea::core::BaselineOptimizer baseline;
        hpoea::core::Budget algo_budget;
        algo_budget.generations = 5u;
        hpoea::core::Budget optimizer_budget;
        optimizer_budget.function_evaluations = 1000u;

        auto result = baseline.optimize(factory, problem, optimizer_budget, algo_budget, 42);

        HPOEA_V2_CHECK(runner, result.optimizer_usage.objective_calls == 1,
                       "baseline optimizer_usage.objective_calls is 1");
        HPOEA_V2_CHECK(runner, result.optimizer_usage.iterations == 0,
                       "baseline optimizer_usage.iterations is 0");
        HPOEA_V2_CHECK(runner, result.optimizer_usage.wall_time.count() >= 0,
                       "baseline optimizer_usage.wall_time is populated");
    }

    return runner.summarize("baseline_optimizer_tests");
}
