#include "test_harness.hpp"

#include "hpoea/core/search_space.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/nm_hyper.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <atomic>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Optimizer>
hpoea::core::HyperparameterOptimizationResult run_optimizer(Optimizer &optimizer,
                                                            hpoea::core::ParameterSet params,
                                                            const hpoea::core::Budget &opt_budget,
                                                            const hpoea::core::Budget &algo_budget,
                                                            unsigned long seed) {
    hpoea::wrappers::problems::SphereProblem problem(4);
    hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    optimizer.configure(params);
    return optimizer.optimize(factory, problem, opt_budget, algo_budget, seed);
}

// controllable inner algorithm
// drives failure paths in the hyper optimizers
enum class FailMode { None, ConfigureThrows, RunFails };

class ControllableAlgorithm final : public hpoea::core::IEvolutionaryAlgorithm {
public:
    ControllableAlgorithm(hpoea::core::ParameterSpace space, FailMode mode,
                          std::shared_ptr<std::atomic<int>> configure_counter, int configure_limit)
        : space_(std::move(space)), mode_(mode),
          configure_counter_(std::move(configure_counter)), configure_limit_(configure_limit) {}

    [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override { return id_; }
    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override { return space_; }

    void configure(const hpoea::core::ParameterSet &) override {
        if (configure_counter_) {
            const int n = configure_counter_->fetch_add(1) + 1;
            if (configure_limit_ >= 0 && n > configure_limit_) {
                throw std::runtime_error("configure limit reached");
            }
        }
        if (mode_ == FailMode::ConfigureThrows) {
            throw std::runtime_error("configure always fails");
        }
    }

    [[nodiscard]] hpoea::core::OptimizationResult run(const hpoea::core::IProblem &problem,
                                                      const hpoea::core::Budget &, unsigned long seed) override {
        hpoea::core::OptimizationResult r;
        r.best_solution.assign(problem.dimension(), 0.0);
        r.algorithm_usage.function_evaluations = 1u;
        r.seed = seed;
        if (mode_ == FailMode::RunFails) {
            r.status = hpoea::core::RunStatus::FailedEvaluation;
            // finite but failed
            // must never be selectable
            r.best_fitness = 3.0;
            r.message = "run failed";
        } else {
            r.status = hpoea::core::RunStatus::Success;
            r.best_fitness = 1.0;
        }
        return r;
    }

    [[nodiscard]] std::unique_ptr<hpoea::core::IEvolutionaryAlgorithm> clone() const override {
        return std::make_unique<ControllableAlgorithm>(*this);
    }

private:
    hpoea::core::AlgorithmIdentity id_{"Controllable", "tests", "1.0"};
    hpoea::core::ParameterSpace space_;
    FailMode mode_;
    std::shared_ptr<std::atomic<int>> configure_counter_;
    int configure_limit_;
};

struct ControllableFactory final : public hpoea::core::IEvolutionaryAlgorithmFactory {
    hpoea::core::ParameterSpace space;
    hpoea::core::AlgorithmIdentity id{"ControllableFactory", "tests", "1.0"};
    FailMode mode{FailMode::None};
    std::shared_ptr<std::atomic<int>> configure_counter;
    int configure_limit{-1};

    [[nodiscard]] hpoea::core::EvolutionaryAlgorithmPtr create() const override {
        return std::make_unique<ControllableAlgorithm>(space, mode, configure_counter, configure_limit);
    }
    [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override { return space; }
    [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override { return id; }
};

hpoea::core::ParameterSpace one_continuous_param_space() {
    hpoea::core::ParameterSpace space;
    hpoea::core::ParameterDescriptor d;
    d.name = "x";
    d.type = hpoea::core::ParameterType::Continuous;
    d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    d.default_value = 0.5;
    space.add_descriptor(d);
    return space;
}

// fix six of the seven DE parameters
// so tuning space has exactly one dimension
std::shared_ptr<hpoea::core::SearchSpace> single_de_dimension() {
    auto search = std::make_shared<hpoea::core::SearchSpace>();
    search->fix("population_size", std::int64_t{20});
    search->fix("generations", std::int64_t{3});
    search->fix("crossover_rate", 0.8);
    search->fix("variant", std::int64_t{2});
    search->fix("ftol", 1e-6);
    search->fix("xtol", 1e-6);
    search->optimize("scaling_factor", hpoea::core::ContinuousRange{0.4, 0.6});
    return search;
}

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    hpoea::core::Budget algo_budget;
    algo_budget.generations = 5u;
    algo_budget.function_evaluations = 500u;

    hpoea::core::Budget opt_budget;
    opt_budget.generations = 5u;
    opt_budget.function_evaluations = 500u;


    {
        // smoke each hyper optimizer
        // runs and records its seed
        // accounts trials
        // reproducible
        struct HoaCase {
            const char *name;
            std::function<hpoea::core::HyperparameterOptimizationResult(
                const hpoea::core::Budget &, const hpoea::core::Budget &, unsigned long)>
                run;
            unsigned long seed;
        };
        const std::vector<HoaCase> hoa_cases = {
            {"CMA-ES hyper",
             [&](const hpoea::core::Budget &ob, const hpoea::core::Budget &ab, unsigned long s) {
                 hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer opt;
                 hpoea::core::ParameterSet p;
                 p.emplace("generations", std::int64_t{5});
                 return run_optimizer(opt, p, ob, ab, s);
             },
             42UL},
            {"PSO hyper",
             [&](const hpoea::core::Budget &ob, const hpoea::core::Budget &ab, unsigned long s) {
                 hpoea::pagmo_wrappers::PagmoPsoHyperOptimizer opt;
                 hpoea::core::ParameterSet p;
                 p.emplace("generations", std::int64_t{5});
                 return run_optimizer(opt, p, ob, ab, s);
             },
             7UL},
            {"SA hyper",
             [&](const hpoea::core::Budget &ob, const hpoea::core::Budget &ab, unsigned long s) {
                 hpoea::pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer opt;
                 hpoea::core::ParameterSet p;
                 p.emplace("iterations", std::int64_t{5});
                 return run_optimizer(opt, p, ob, ab, s);
             },
             11UL},
            {"NM hyper",
             [&](const hpoea::core::Budget &ob, const hpoea::core::Budget &ab, unsigned long s) {
                 hpoea::pagmo_wrappers::PagmoNelderMeadHyperOptimizer opt;
                 hpoea::core::ParameterSet p;
                 p.emplace("max_fevals", std::int64_t{50});
                 return run_optimizer(opt, p, ob, ab, s);
             },
             13UL},
        };
        for (const auto &c : hoa_cases) {
            const std::string name = c.name;
            const auto result = c.run(opt_budget, algo_budget, c.seed);
            HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success ||
                                      result.status == hpoea::core::RunStatus::BudgetExceeded,
                           name + " returns success/budget");
            HPOEA_V2_CHECK(runner, !result.trials.empty(), name + " produces trials");
            HPOEA_V2_CHECK(runner, !result.best_parameters.empty(),
                           name + " best_parameters populated");
            HPOEA_V2_CHECK(runner, std::isfinite(result.best_objective),
                           name + " best_objective finite");
            HPOEA_V2_CHECK(runner, result.seed == c.seed, name + " records optimizer seed");
            HPOEA_V2_CHECK(runner, result.optimizer_usage.objective_calls == result.trials.size(),
                           name + " objective_calls matches recorded trials");

            const auto result_same = c.run(opt_budget, algo_budget, c.seed);
            HPOEA_V2_CHECK(runner, result.best_objective == result_same.best_objective,
                           name + " deterministic for same seed");
        }
    }


    {
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        auto search = std::make_shared<hpoea::core::SearchSpace>();
        search->fix("population_size", std::int64_t{40});
        search->optimize("scaling_factor", hpoea::core::ContinuousRange{0.6, 0.7});
        optimizer.set_search_space(search);

        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{5});
        auto result = run_optimizer(optimizer, params, opt_budget, algo_budget, 42UL);
        bool all_fixed = true;
        bool all_bounds = true;
        for (const auto &trial : result.trials) {
            const auto pop = std::get<std::int64_t>(trial.parameters.at("population_size"));
            const auto sf = std::get<double>(trial.parameters.at("scaling_factor"));
            if (pop != 40) {
                all_fixed = false;
            }
            if (sf < 0.6 || sf > 0.7) {
                all_bounds = false;
            }
        }
        HPOEA_V2_CHECK(runner, all_fixed, "search space fixed parameter enforced");
        HPOEA_V2_CHECK(runner, all_bounds, "search space bounds enforced");
    }


    {
        // unknown fixed key and fully-fixed space both leave nothing valid to tune
        auto status_for = [&](const std::shared_ptr<hpoea::core::SearchSpace> &search) {
            hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
            optimizer.set_search_space(search);
            hpoea::core::ParameterSet params;
            params.emplace("generations", std::int64_t{5});
            return run_optimizer(optimizer, params, opt_budget, algo_budget, 1UL).status;
        };

        auto unknown = std::make_shared<hpoea::core::SearchSpace>();
        unknown->fix("unknown_param", 1.0);
        HPOEA_V2_CHECK(runner, status_for(unknown) == hpoea::core::RunStatus::InvalidConfiguration,
                       "invalid search space yields InvalidConfiguration");

        auto all_fixed = std::make_shared<hpoea::core::SearchSpace>();
        all_fixed->fix("population_size", std::int64_t{20});
        all_fixed->fix("generations", std::int64_t{5});
        all_fixed->fix("scaling_factor", 0.6);
        all_fixed->fix("crossover_rate", 0.8);
        all_fixed->fix("variant", std::int64_t{2});
        all_fixed->fix("ftol", 1e-6);
        all_fixed->fix("xtol", 1e-6);
        HPOEA_V2_CHECK(runner, status_for(all_fixed) == hpoea::core::RunStatus::InvalidConfiguration,
                       "fully fixed search space yields InvalidConfiguration");
    }


    {
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{2});
        hpoea::core::Budget small_opt_budget;
        small_opt_budget.generations = 1u;
        auto result = run_optimizer(optimizer, params, small_opt_budget, algo_budget, 42UL);
        HPOEA_V2_CHECK(runner, result.optimizer_usage.iterations <= 1u,
                       "optimizer budget iterations enforced");
        const auto eff = result.effective_optimizer_parameters.find("generations");
        HPOEA_V2_CHECK(runner, eff != result.effective_optimizer_parameters.end() &&
                                  std::get<std::int64_t>(eff->second) == 1,
                       "CMA-ES effective_optimizer_parameters stamps clamped generations (1), not raw (2)");
    }


    {
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        optimizer.set_search_space(single_de_dimension());
        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{3});
        auto result = run_optimizer(optimizer, params, opt_budget, algo_budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success ||
                                  result.status == hpoea::core::RunStatus::BudgetExceeded,
                       "CMA-ES one-parameter tuning space completes without invalid_argument (pop_size >= 5)");
        HPOEA_V2_CHECK(runner, !result.trials.empty() && std::isfinite(result.best_objective),
                       "CMA-ES one-parameter tuning space produces a finite result");
    }


    {
        // dim=1, n_T_adj=10, n_range_adj=1, bin_size=10
        // so 100
        const std::size_t evals_per_evolve = 100u;

        hpoea::core::ParameterSet params;
        params.emplace("iterations", std::int64_t{1000});

        hpoea::pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer optimizer;
        optimizer.set_search_space(single_de_dimension());

        hpoea::core::Budget exact_budget;
        exact_budget.function_evaluations = evals_per_evolve + 1;
        auto exact = run_optimizer(optimizer, params, exact_budget, algo_budget, 3UL);
        HPOEA_V2_CHECK(runner, exact.status == hpoea::core::RunStatus::Success,
                       "SA budget evals_per_evolve+1 ends Success (initial evaluation reserved)");
        HPOEA_V2_CHECK(runner, exact.optimizer_usage.objective_calls == evals_per_evolve + 1,
                       "SA spends exactly the budget in objective_calls, no overshoot by one");
        HPOEA_V2_CHECK(runner, exact.optimizer_usage.iterations == 1u,
                       "SA performs exactly one evolve at evals_per_evolve+1 budget");

        hpoea::core::Budget starved_budget;
        starved_budget.function_evaluations = evals_per_evolve;
        auto starved = run_optimizer(optimizer, params, starved_budget, algo_budget, 3UL);
        HPOEA_V2_CHECK(runner, starved.status == hpoea::core::RunStatus::BudgetExceeded,
                       "SA zero-evolve budget reports BudgetExceeded, not Success");
        HPOEA_V2_CHECK(runner, starved.optimizer_usage.iterations == 0u,
                       "SA zero-evolve budget performs zero evolves");
        HPOEA_V2_CHECK(runner, starved.message.find("insufficient") != std::string::npos &&
                                  starved.message.find("101") != std::string::npos,
                       "SA zero-evolve message states insufficient budget and minimum 1 + evals_per_evolve");
    }


    {
        // default DE tuning space has 7 dimensions
        // so nlopt simplex is pop = dim + 1 = 8
        // NM reserves pop + 1 (final re-evaluation)
        // a bound run spends exactly the budget
        hpoea::pagmo_wrappers::PagmoNelderMeadHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        // large so the budget binds
        // not the config
        params.emplace("max_fevals", std::int64_t{1000});

        hpoea::core::Budget exact_budget;
        exact_budget.function_evaluations = 50u;
        auto exact = run_optimizer(optimizer, params, exact_budget, algo_budget, 3UL);
        HPOEA_V2_CHECK(runner, exact.status == hpoea::core::RunStatus::Success,
                       "NM budget above the simplex+overhead ends Success (final re-evaluation reserved)");
        HPOEA_V2_CHECK(runner, exact.optimizer_usage.objective_calls == 50u,
                       "NM spends exactly the budget in objective_calls, no overshoot by one");
        HPOEA_V2_CHECK(runner, exact.optimizer_usage.iterations == 1u,
                       "NM performs exactly one nelder-mead evolve within budget");

        hpoea::core::Budget starved_budget;
        // pop(8) + 1
        // nothing left for a nelder-mead step
        starved_budget.function_evaluations = 9u;
        auto starved = run_optimizer(optimizer, params, starved_budget, algo_budget, 3UL);
        HPOEA_V2_CHECK(runner, starved.status == hpoea::core::RunStatus::BudgetExceeded,
                       "NM budget <= pop+1 reports BudgetExceeded, not Success");
        HPOEA_V2_CHECK(runner, starved.optimizer_usage.iterations == 0u,
                       "NM starved budget performs zero evolves");
        HPOEA_V2_CHECK(runner, starved.message.find("insufficient") != std::string::npos &&
                                  starved.message.find("10") != std::string::npos,
                       "NM starved message states insufficient budget and the minimum evaluation count");
    }


    {
        hpoea::pagmo_wrappers::PagmoPsoHyperOptimizer optimizer;
        const auto &desc = optimizer.parameter_space().descriptor("max_velocity");
        HPOEA_V2_CHECK(runner, desc.continuous_range.has_value(),
                       "PSO max_velocity descriptor has a continuous range");
        if (desc.continuous_range.has_value()) {
            HPOEA_V2_CHECK(runner, desc.continuous_range->lower == 0.01 &&
                                      desc.continuous_range->upper == 1.0,
                           "PSO max_velocity descriptor range is [0.01, 1.0], within pagmo's (0, 1]");
        }
    }


    {
        ControllableFactory factory;
        factory.space = one_continuous_param_space();
        factory.mode = FailMode::ConfigureThrows;

        hpoea::wrappers::problems::SphereProblem problem(4);
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{3});
        optimizer.configure(params);
        auto result = optimizer.optimize(factory, problem, opt_budget, algo_budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status != hpoea::core::RunStatus::Success,
                       "hyper run whose every configure throws is not Success");
        HPOEA_V2_CHECK(runner, std::isinf(result.best_objective),
                       "hyper run whose every configure throws reports infinite best_objective");
    }


    {
        ControllableFactory factory;
        factory.space = one_continuous_param_space();
        factory.mode = FailMode::RunFails;

        hpoea::wrappers::problems::SphereProblem problem(4);
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{2});
        optimizer.configure(params);
        auto result = optimizer.optimize(factory, problem, opt_budget, algo_budget, 42UL);
        HPOEA_V2_CHECK(runner, !result.trials.empty(),
                       "all-failed run still records trials");
        HPOEA_V2_CHECK(runner, result.status != hpoea::core::RunStatus::Success,
                       "all-failed run (finite failed trials) is not Success");
        HPOEA_V2_CHECK(runner, std::isinf(result.best_objective),
                       "all-failed run reports infinite best_objective");
        HPOEA_V2_CHECK(runner, result.best_parameters.empty(),
                       "all-failed run leaves best_parameters empty (no selectable trial)");
    }


    {
        const int k = 3;
        ControllableFactory factory;
        factory.space = one_continuous_param_space();
        factory.mode = FailMode::None;
        factory.configure_counter = std::make_shared<std::atomic<int>>(0);
        // (k+1)-th configure throws before its trial is counted
        factory.configure_limit = k;

        hpoea::wrappers::problems::SphereProblem problem(4);
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{3});
        optimizer.configure(params);
        auto result = optimizer.optimize(factory, problem, opt_budget, algo_budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status != hpoea::core::RunStatus::Success,
                       "hyper run that dies mid-flight is not Success");
        HPOEA_V2_CHECK(runner, result.optimizer_usage.objective_calls == static_cast<std::size_t>(k),
                       "mid-flight failure reports objective_calls equal to completed trials (k), not 0");
        HPOEA_V2_CHECK(runner, result.trials.size() == static_cast<std::size_t>(k),
                       "mid-flight failure recovers the k completed trials");
    }

    return runner.summarize("hyper_optimizer_tests");
}
