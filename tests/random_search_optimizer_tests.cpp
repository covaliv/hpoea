#include "test_harness.hpp"
#include "test_fixtures.hpp"
#include "test_utils.hpp"

#include "hpoea/core/random_search_optimizer.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Observation {
    hpoea::core::ParameterSet parameters;
    unsigned long seed{0};
};

using RunFn = std::function<hpoea::core::OptimizationResult(
    const hpoea::core::ParameterSet &, const hpoea::core::IProblem &, const hpoea::core::Budget &, unsigned long)>;

hpoea::core::ParameterSpace make_algorithm_space() {
    hpoea::core::ParameterSpace space;

    hpoea::core::ParameterDescriptor d;
    d.name = "rate";
    d.type = hpoea::core::ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.01, 100.0};
    d.default_value = 1.0;
    space.add_descriptor(d);

    d = {};
    d.name = "population";
    d.type = hpoea::core::ParameterType::Integer;
    d.integer_range = hpoea::core::IntegerRange{1, 10};
    d.default_value = std::int64_t{5};
    space.add_descriptor(d);

    d = {};
    d.name = "flag";
    d.type = hpoea::core::ParameterType::Boolean;
    d.default_value = false;
    space.add_descriptor(d);

    d = {};
    d.name = "category";
    d.type = hpoea::core::ParameterType::Categorical;
    d.categorical_choices = {"a", "b", "c"};
    d.default_value = std::string{"a"};
    space.add_descriptor(d);

    return space;
}

hpoea::core::OptimizationResult successful_result(const hpoea::core::ParameterSet &parameters,
                                                  const hpoea::core::IProblem &problem,
                                                  const hpoea::core::Budget &budget,
                                                  unsigned long seed) {
    const auto rate = std::get<double>(parameters.at("rate"));
    const auto population = static_cast<double>(std::get<std::int64_t>(parameters.at("population")));
    const auto flag = std::get<bool>(parameters.at("flag")) ? 0.5 : 0.0;
    const auto category = std::get<std::string>(parameters.at("category")) == "c" ? 0.25 : 0.0;

    hpoea::core::OptimizationResult result;
    result.status = hpoea::core::RunStatus::Success;
    result.seed = seed;
    result.best_fitness = rate + population + flag + category;
    result.best_solution = std::vector<double>(problem.dimension(), result.best_fitness);
    result.requested_budget = budget;
    result.effective_budget = budget;
    result.algorithm_usage.function_evaluations = 1;
    result.effective_parameters = parameters;
    result.message = "fake run completed";
    return result;
}

hpoea::core::OptimizationResult failed_result(unsigned long seed) {
    hpoea::core::OptimizationResult result;
    result.status = hpoea::core::RunStatus::FailedEvaluation;
    result.seed = seed;
    result.best_fitness = std::numeric_limits<double>::infinity();
    result.error_info = hpoea::core::ErrorInfo{"evaluation_failure", "test", "scripted failure"};
    result.message = "scripted failure";
    return result;
}

class FakeAlgorithm final : public hpoea::core::IEvolutionaryAlgorithm {
  public:
    FakeAlgorithm(hpoea::core::ParameterSpace space, std::shared_ptr<std::vector<Observation>> observations,
                  RunFn run_fn)
        : space_(std::move(space)), observations_(std::move(observations)), run_fn_(std::move(run_fn)) {}

    [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override {
        return identity_;
    }

    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
        return space_;
    }

    void configure(const hpoea::core::ParameterSet &parameters) override {
        configured_ = space_.apply_defaults(parameters);
        space_.validate(configured_);
    }

    [[nodiscard]] hpoea::core::OptimizationResult run(const hpoea::core::IProblem &problem,
                                                      const hpoea::core::Budget &budget, unsigned long seed) override {
        observations_->push_back(Observation{configured_, seed});
        if (run_fn_) {
            return run_fn_(configured_, problem, budget, seed);
        }
        return successful_result(configured_, problem, budget, seed);
    }

    [[nodiscard]] hpoea::core::EvolutionaryAlgorithmPtr clone() const override {
        return std::make_unique<FakeAlgorithm>(*this);
    }

  private:
    hpoea::core::AlgorithmIdentity identity_{"FakeAlgorithm", "tests", "1.0"};
    hpoea::core::ParameterSpace space_;
    hpoea::core::ParameterSet configured_;
    std::shared_ptr<std::vector<Observation>> observations_;
    RunFn run_fn_;
};

class FakeFactory final : public hpoea::core::IEvolutionaryAlgorithmFactory {
  public:
    explicit FakeFactory(RunFn run_fn = {}) : run_fn_(std::move(run_fn)) {}

    [[nodiscard]] hpoea::core::EvolutionaryAlgorithmPtr create() const override {
        return std::make_unique<FakeAlgorithm>(space_, observations_, run_fn_);
    }

    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
        return space_;
    }

    [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override {
        return identity_;
    }

    [[nodiscard]] const std::vector<Observation> &observations() const noexcept {
        return *observations_;
    }

  private:
    hpoea::core::ParameterSpace space_{make_algorithm_space()};
    hpoea::core::AlgorithmIdentity identity_{"FakeFactory", "tests", "1.0"};
    std::shared_ptr<std::vector<Observation>> observations_{std::make_shared<std::vector<Observation>>()};
    RunFn run_fn_;
};

class EmptyFactory final : public hpoea::core::IEvolutionaryAlgorithmFactory {
  public:
    [[nodiscard]] hpoea::core::EvolutionaryAlgorithmPtr create() const override {
        throw std::runtime_error("empty factory should not create algorithms");
    }

    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
        return space_;
    }

    [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override {
        return identity_;
    }

  private:
    hpoea::core::ParameterSpace space_;
    hpoea::core::AlgorithmIdentity identity_{"EmptyFactory", "tests", "1.0"};
};

class ThrowingFactory final : public hpoea::core::IEvolutionaryAlgorithmFactory {
  public:
    [[nodiscard]] hpoea::core::EvolutionaryAlgorithmPtr create() const override {
        throw std::runtime_error("factory create failed");
    }

    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
        return space_;
    }

    [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override {
        return identity_;
    }

  private:
    hpoea::core::ParameterSpace space_{make_algorithm_space()};
    hpoea::core::AlgorithmIdentity identity_{"ThrowingFactory", "tests", "1.0"};
};

void configure_samples(hpoea::core::RandomSearchOptimizer &optimizer, std::int64_t sample_count) {
    hpoea::core::ParameterSet params;
    params.emplace("sample_count", sample_count);
    optimizer.configure(params);
}

bool category_is_b_or_c(const hpoea::core::ParameterValue &value) {
    if (!std::holds_alternative<std::string>(value)) {
        return false;
    }
    const auto &label = std::get<std::string>(value);
    return label == "b" || label == "c";
}

bool trial_parameters_equal(const std::vector<hpoea::core::HyperparameterTrialRecord> &lhs,
                            const std::vector<hpoea::core::HyperparameterTrialRecord> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (!hpoea::tests_v2::parameter_set_equals(lhs[i].parameters, rhs[i].parameters)) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    hpoea::tests_v2::TestRunner runner;
    hpoea::tests_v2::DummyProblem problem(2);


    {
        hpoea::core::RandomSearchOptimizer optimizer;
        HPOEA_V2_CHECK(runner, optimizer.identity().family == "RandomSearch", "identity family is RandomSearch");
        HPOEA_V2_CHECK(runner, optimizer.identity().implementation == "uniform_random",
                       "identity implementation is uniform_random");
        HPOEA_V2_CHECK(runner, optimizer.parameter_space().contains("sample_count"),
                       "random search exposes sample_count");

        FakeFactory factory;
        auto result = optimizer.optimize(factory, problem, {}, {}, 123UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "default random search succeeds");
        HPOEA_V2_CHECK(runner, result.trials.size() == 100u, "default sample_count runs 100 trials");
        HPOEA_V2_CHECK(runner, result.optimizer_usage.objective_calls == 100u,
                       "default objective_calls matches trials");

        bool zero_rejected = false;
        try {
            configure_samples(optimizer, 0);
        } catch (const hpoea::core::ParameterValidationError &) {
            zero_rejected = true;
        }
        HPOEA_V2_CHECK(runner, zero_rejected, "sample_count=0 is rejected");
    }


    {
        hpoea::core::RandomSearchOptimizer optimizer;
        configure_samples(optimizer, 5);

        auto search = std::make_shared<hpoea::core::SearchSpace>();
        search->optimize("rate", hpoea::core::ContinuousRange{0.1, 10.0}, hpoea::core::Transform::log);
        search->optimize_choices("population", {std::int64_t{3}, std::int64_t{7}});
        search->exclude("flag");
        search->optimize_choices("category", {std::string{"b"}, std::string{"c"}});
        optimizer.set_search_space(search);

        FakeFactory factory;
        auto result = optimizer.optimize(factory, problem, {}, {}, 77UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "constrained random search succeeds");
        HPOEA_V2_CHECK(runner, result.trials.size() == 5u, "configured sample_count controls trials");
        HPOEA_V2_CHECK(runner, result.optimizer_usage.objective_calls == 5u,
                       "objective_calls matches completed runs");
        HPOEA_V2_CHECK(runner, result.optimizer_usage.iterations == 5u, "iterations matches samples");

        bool all_valid = true;
        double best = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < result.trials.size(); ++i) {
            const auto &trial = result.trials[i];
            const auto rate = std::get<double>(trial.parameters.at("rate"));
            const auto population = std::get<std::int64_t>(trial.parameters.at("population"));
            all_valid = all_valid && rate >= 0.1 && rate <= 10.0;
            all_valid = all_valid && (population == 3 || population == 7);
            all_valid = all_valid && !trial.parameters.contains("flag");
            all_valid = all_valid && category_is_b_or_c(trial.parameters.at("category"));
            all_valid = all_valid && trial.trial_index == i;
            best = std::min(best, trial.optimization_result.best_fitness);
        }
        HPOEA_V2_CHECK(runner, all_valid, "samples respect bounds, choices, excludes, and indexes");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(result.best_objective, best),
                       "best objective is selected from sampled trials");

        hpoea::core::RandomSearchOptimizer repeat;
        configure_samples(repeat, 5);
        repeat.set_search_space(std::make_shared<hpoea::core::SearchSpace>(*search));
        FakeFactory repeat_factory;
        auto repeat_result = repeat.optimize(repeat_factory, problem, {}, {}, 77UL);
        HPOEA_V2_CHECK(runner, trial_parameters_equal(result.trials, repeat_result.trials),
                       "same seed reproduces sampled parameters");
        bool same_inner_seeds = factory.observations().size() == repeat_factory.observations().size();
        for (std::size_t i = 0; same_inner_seeds && i < factory.observations().size(); ++i) {
            same_inner_seeds = factory.observations()[i].seed == repeat_factory.observations()[i].seed;
        }
        HPOEA_V2_CHECK(runner, same_inner_seeds, "same seed reproduces derived inner seeds");
    }


    {
        hpoea::core::RandomSearchOptimizer optimizer;
        configure_samples(optimizer, 6);
        FakeFactory factory;

        hpoea::core::Budget budget;
        budget.function_evaluations = 4u;
        budget.generations = 2u;
        auto capped = optimizer.optimize(factory, problem, budget, {}, 7UL);
        HPOEA_V2_CHECK(runner, capped.status == hpoea::core::RunStatus::Success,
                       "capped random search can complete successfully");
        HPOEA_V2_CHECK(runner, capped.trials.size() == 2u, "tightest optimizer budget caps samples");
        HPOEA_V2_CHECK(runner, capped.optimizer_usage.objective_calls == 2u,
                       "capped objective_calls matches samples");

        budget.generations = 0u;
        auto zero = optimizer.optimize(factory, problem, budget, {}, 7UL);
        HPOEA_V2_CHECK(runner, zero.status == hpoea::core::RunStatus::BudgetExceeded,
                       "zero optimizer budget returns BudgetExceeded");
        HPOEA_V2_CHECK(runner, zero.trials.empty(), "zero optimizer budget records no trials");

        hpoea::core::Budget wall_time;
        wall_time.wall_time = std::chrono::milliseconds{0};
        auto timed_out = optimizer.optimize(factory, problem, wall_time, {}, 7UL);
        HPOEA_V2_CHECK(runner, timed_out.status == hpoea::core::RunStatus::BudgetExceeded,
                       "zero wall-time budget returns BudgetExceeded");
        HPOEA_V2_CHECK(runner, timed_out.trials.empty(), "zero wall-time budget records no trials");
        HPOEA_V2_CHECK(runner, timed_out.optimizer_usage.objective_calls == 0u,
                       "zero wall-time budget skips objective calls");
    }


    {
        hpoea::core::RandomSearchOptimizer optimizer;
        auto fixed = std::make_shared<hpoea::core::SearchSpace>();
        fixed->fix("rate", 1.0);
        fixed->fix("population", std::int64_t{5});
        fixed->fix("flag", false);
        fixed->fix("category", std::string{"a"});
        optimizer.set_search_space(fixed);
        FakeFactory factory;
        auto fixed_result = optimizer.optimize(factory, problem, {}, {}, 1UL);
        HPOEA_V2_CHECK(runner, fixed_result.status == hpoea::core::RunStatus::InvalidConfiguration,
                       "fully fixed search space is InvalidConfiguration");
        HPOEA_V2_CHECK(runner, fixed_result.trials.empty(), "fully fixed search space records no trials");

        auto unknown = std::make_shared<hpoea::core::SearchSpace>();
        unknown->fix("unknown", 1.0);
        optimizer.set_search_space(unknown);
        auto unknown_result = optimizer.optimize(factory, problem, {}, {}, 1UL);
        HPOEA_V2_CHECK(runner, unknown_result.status == hpoea::core::RunStatus::InvalidConfiguration,
                       "unknown search-space parameter is InvalidConfiguration");
        HPOEA_V2_CHECK(runner, unknown_result.trials.empty(), "invalid search space records no trials");

        EmptyFactory empty_factory;
        hpoea::core::RandomSearchOptimizer empty_optimizer;
        auto empty_result = empty_optimizer.optimize(empty_factory, problem, {}, {}, 1UL);
        HPOEA_V2_CHECK(runner, empty_result.status == hpoea::core::RunStatus::InvalidConfiguration,
                       "empty algorithm parameter space is InvalidConfiguration");
    }


    {
        auto counter = std::make_shared<std::size_t>(0);
        FakeFactory mixed_factory{[counter](const hpoea::core::ParameterSet &parameters,
                                            const hpoea::core::IProblem &problem,
                                            const hpoea::core::Budget &budget, unsigned long seed) {
            if ((*counter)++ == 0u) {
                auto result = successful_result(parameters, problem, budget, seed);
                result.best_fitness = std::numeric_limits<double>::infinity();
                return result;
            }

            hpoea::core::OptimizationResult result;
            result.status = hpoea::core::RunStatus::BudgetExceeded;
            result.seed = seed;
            result.best_fitness = 0.25;
            result.message = "inner budget exceeded with finite result";
            return result;
        }};

        hpoea::core::RandomSearchOptimizer optimizer;
        configure_samples(optimizer, 2);
        auto mixed = optimizer.optimize(mixed_factory, problem, {}, {}, 99UL);
        HPOEA_V2_CHECK(runner, mixed.status == hpoea::core::RunStatus::Success,
                       "finite later trial makes optimizer succeed");
        HPOEA_V2_CHECK(runner, mixed.trials[0].optimization_result.status == hpoea::core::RunStatus::InternalError,
                       "non-finite successful trial becomes InternalError");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(mixed.best_objective, 0.25),
                       "finite inner BudgetExceeded trial is selectable as best");

        FakeFactory failing_factory{[](const hpoea::core::ParameterSet &, const hpoea::core::IProblem &,
                                       const hpoea::core::Budget &, unsigned long seed) {
            return failed_result(seed);
        }};
        auto failing = optimizer.optimize(failing_factory, problem, {}, {}, 99UL);
        HPOEA_V2_CHECK(runner, failing.status == hpoea::core::RunStatus::FailedEvaluation,
                       "all failed trials propagate failure status");
        HPOEA_V2_CHECK(runner, failing.error_info.has_value() && failing.error_info->category == "evaluation_failure",
                       "all failed trials preserve error_info");

        ThrowingFactory throwing_factory;
        auto thrown = optimizer.optimize(throwing_factory, problem, {}, {}, 99UL);
        HPOEA_V2_CHECK(runner, thrown.status == hpoea::core::RunStatus::InternalError,
                       "factory create failure maps to InternalError");
        HPOEA_V2_CHECK(runner, thrown.trials.size() == 2u, "factory create failures record sampled trials");
        HPOEA_V2_CHECK(runner, thrown.optimizer_usage.objective_calls == 0u,
                       "objective_calls excludes failures before run");
        HPOEA_V2_CHECK(runner, thrown.optimizer_usage.iterations == 2u,
                       "iterations counts sampled candidates");
    }


    {
        hpoea::core::RandomSearchOptimizer optimizer;
        hpoea::core::ExperimentConfig config;
        config.experiment_id = "random_search_experiment";
        config.trials_per_optimizer = 1;
        config.random_seed = 123UL;
        config.optimizer_parameters = hpoea::core::ParameterSet{{"sample_count", std::int64_t{2}}};

        FakeFactory factory;
        hpoea::tests_v2::CapturingLogger logger;
        hpoea::core::SequentialExperimentManager manager;
        auto result = manager.run_experiment(config, optimizer, factory, problem, logger);

        HPOEA_V2_CHECK(runner, result.optimizer_results.size() == 1u,
                       "experiment records one optimizer result");
        HPOEA_V2_CHECK(runner, logger.records.size() == 2u,
                       "experiment logs one record per random search sample");
        HPOEA_V2_CHECK(runner, logger.records.front().hyper_optimizer.has_value() &&
                                  logger.records.front().hyper_optimizer->family == "RandomSearch",
                       "logged records identify RandomSearch");
        HPOEA_V2_CHECK(runner, logger.records.front().optimizer_parameters.contains("sample_count"),
                       "logged records include random search parameters");
    }

    return runner.summarize("random_search_optimizer_tests");
}
