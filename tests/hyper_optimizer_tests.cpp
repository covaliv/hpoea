#include "test_harness.hpp"

#include "hpoea/core/search_space.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/nm_hyper.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>

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
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{5});
        auto result = run_optimizer(optimizer, params, opt_budget, algo_budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success ||
                                  result.status == hpoea::core::RunStatus::BudgetExceeded,
                       "CMA-ES hyper returns success/budget");
        HPOEA_V2_CHECK(runner, !result.trials.empty(), "CMA-ES hyper produces trials");
        HPOEA_V2_CHECK(runner, !result.best_parameters.empty(), "CMA-ES hyper best_parameters populated");
        HPOEA_V2_CHECK(runner, std::isfinite(result.best_objective), "CMA-ES hyper best_objective finite");

        auto result_same = run_optimizer(optimizer, params, opt_budget, algo_budget, 42UL);
        HPOEA_V2_CHECK(runner, result.best_objective == result_same.best_objective,
                       "CMA-ES hyper deterministic for same seed");

        auto result_diff = run_optimizer(optimizer, params, opt_budget, algo_budget, 43UL);
        HPOEA_V2_CHECK(runner, result.best_objective != result_diff.best_objective ||
                                  result.trials.size() != result_diff.trials.size(),
                       "CMA-ES hyper different seed changes result");
    }


    {
        hpoea::pagmo_wrappers::PagmoPsoHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{5});
        auto result = run_optimizer(optimizer, params, opt_budget, algo_budget, 7UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success ||
                                  result.status == hpoea::core::RunStatus::BudgetExceeded,
                       "PSO hyper returns success/budget");
        HPOEA_V2_CHECK(runner, !result.trials.empty(), "PSO hyper produces trials");
        HPOEA_V2_CHECK(runner, !result.best_parameters.empty(), "PSO hyper best_parameters populated");

        auto result_same = run_optimizer(optimizer, params, opt_budget, algo_budget, 7UL);
        HPOEA_V2_CHECK(runner, result.best_objective == result_same.best_objective,
                       "PSO hyper deterministic for same seed");
    }


    {
        hpoea::pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        params.emplace("iterations", std::int64_t{5});
        auto result = run_optimizer(optimizer, params, opt_budget, algo_budget, 11UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success ||
                                  result.status == hpoea::core::RunStatus::BudgetExceeded,
                       "SA hyper returns success/budget");
        HPOEA_V2_CHECK(runner, !result.trials.empty(), "SA hyper produces trials");
        HPOEA_V2_CHECK(runner, !result.best_parameters.empty(), "SA hyper best_parameters populated");

        auto result_same = run_optimizer(optimizer, params, opt_budget, algo_budget, 11UL);
        HPOEA_V2_CHECK(runner, result.best_objective == result_same.best_objective,
                       "SA hyper deterministic for same seed");
    }


    {
        hpoea::pagmo_wrappers::PagmoNelderMeadHyperOptimizer optimizer;
        hpoea::core::ParameterSet params;
        params.emplace("max_fevals", std::int64_t{50});
        auto result = run_optimizer(optimizer, params, opt_budget, algo_budget, 13UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success ||
                                  result.status == hpoea::core::RunStatus::BudgetExceeded,
                       "NM hyper returns success/budget");
        HPOEA_V2_CHECK(runner, !result.trials.empty(), "NM hyper produces trials");
        HPOEA_V2_CHECK(runner, !result.best_parameters.empty(), "NM hyper best_parameters populated");

        auto result_same = run_optimizer(optimizer, params, opt_budget, algo_budget, 13UL);
        HPOEA_V2_CHECK(runner, result.best_objective == result_same.best_objective,
                       "NM hyper deterministic for same seed");
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
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        auto search = std::make_shared<hpoea::core::SearchSpace>();
        search->fix("unknown_param", 1.0);
        optimizer.set_search_space(search);

        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{5});
        auto result = run_optimizer(optimizer, params, opt_budget, algo_budget, 1UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::InvalidConfiguration,
                       "invalid search space yields InvalidConfiguration");
    }


    {
        hpoea::pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
        auto search = std::make_shared<hpoea::core::SearchSpace>();
        search->fix("population_size", std::int64_t{20});
        search->fix("generations", std::int64_t{5});
        search->fix("scaling_factor", 0.6);
        search->fix("crossover_rate", 0.8);
        search->fix("variant", std::int64_t{2});
        search->fix("ftol", 1e-6);
        search->fix("xtol", 1e-6);
        optimizer.set_search_space(search);

        hpoea::core::ParameterSet params;
        params.emplace("generations", std::int64_t{5});
        auto result = run_optimizer(optimizer, params, opt_budget, algo_budget, 5UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::InvalidConfiguration,
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
    }

    return runner.summarize("hyper_optimizer_tests");
}
