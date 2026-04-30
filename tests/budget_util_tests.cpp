#include "test_harness.hpp"

#include "hpoea/core/budget_checks.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/wrappers/pagmo/cmaes_algorithm.hpp"
#include "hpoea/wrappers/pagmo/de1220_algorithm.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sga_algorithm.hpp"

#include "budget_util.hpp"

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>

using hpoea::core::Budget;
using hpoea::core::AlgorithmRunUsage;
using hpoea::core::OptimizerRunUsage;
using hpoea::core::ParameterSet;
using hpoea::core::RunStatus;

namespace {

bool contains_all(const std::string &text, std::initializer_list<const char *> needles) {
    for (const auto *needle : needles) {
        if (text.find(needle) == std::string::npos) {
            return false;
        }
    }
    return true;
}

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    auto check_algorithm_budget_status = [&](const Budget &budget,
                                             const AlgorithmRunUsage &usage,
                                             RunStatus initial_status,
                                             const std::string &initial_message,
                                             RunStatus expected_status,
                                             const std::string &expected_message,
                                             const std::string &label) {
        auto status = initial_status;
        std::string message = initial_message;
        hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == expected_status, label + ": status");
        HPOEA_V2_CHECK(runner, message == expected_message, label + ": message");
    };

    auto check_optimizer_budget_status = [&](const Budget &budget,
                                             const OptimizerRunUsage &usage,
                                             RunStatus initial_status,
                                             const std::string &initial_message,
                                             RunStatus expected_status,
                                             const std::string &expected_message,
                                             const std::string &label) {
        auto status = initial_status;
        std::string message = initial_message;
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == expected_status, label + ": status");
        HPOEA_V2_CHECK(runner, message == expected_message, label + ": message");
    };


    {
        ParameterSet params;
        params.emplace("generations", std::int64_t{50});
        Budget budget;
        budget.generations = 10u;
        const std::size_t gens = hpoea::pagmo_wrappers::compute_generations(params, budget, 20);
        HPOEA_V2_CHECK(runner, gens == 10u, "compute_generations respects generation budget");
    }

    {
        ParameterSet params;
        params.emplace("generations", std::int64_t{50});
        Budget budget;
        budget.function_evaluations = 100u;
        const std::size_t gens = hpoea::pagmo_wrappers::compute_generations(params, budget, 20);
        HPOEA_V2_CHECK(runner, gens == 4u, "compute_generations respects feval budget");
    }

    {
        ParameterSet params;
        params.emplace("generations", std::int64_t{0});
        Budget budget;
        bool threw = false;
        try {
            (void)hpoea::pagmo_wrappers::compute_generations(params, budget, 20);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "compute_generations rejects zero generations param");
    }

    {
        ParameterSet params;
        params.emplace("generations", std::int64_t{5});
        Budget budget;
        budget.function_evaluations = 10u;
        const std::size_t gens = hpoea::pagmo_wrappers::compute_generations(params, budget, 20);
        HPOEA_V2_CHECK(runner, gens == 0u, "compute_generations returns zero when fevals < pop");
    }


    {
        const unsigned long seed = std::numeric_limits<unsigned long>::max();
        const unsigned a = hpoea::pagmo_wrappers::to_seed32(seed);
        const unsigned b = hpoea::pagmo_wrappers::to_seed32(seed);
        HPOEA_V2_CHECK(runner, a == b,
                       "to_seed32 is deterministic");

        const unsigned c = hpoea::pagmo_wrappers::to_seed32(123UL);
        HPOEA_V2_CHECK(runner, a != c,
                       "to_seed32 produces different values for different seeds");
    }


    {
        const unsigned a = hpoea::pagmo_wrappers::derive_seed(42UL, 1UL);
        const unsigned b = hpoea::pagmo_wrappers::derive_seed(42UL, 1UL);
        const unsigned c = hpoea::pagmo_wrappers::derive_seed(42UL, 2UL);
        HPOEA_V2_CHECK(runner, a == b, "derive_seed deterministic");
        HPOEA_V2_CHECK(runner, a != c, "derive_seed changes with salt");
    }


    {
        hpoea::wrappers::problems::SphereProblem problem(3);
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        auto algo = factory.create();
        ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        algo->configure(params);

        Budget budget;
        budget.generations = 5u;
        auto result = algo->run(problem, budget, 42UL);

        const std::size_t fevals = result.algorithm_usage.function_evaluations;
        const std::size_t expected_gens = fevals > 20u ? (fevals - 20u) / 20u : 0u;
        HPOEA_V2_CHECK(runner, result.algorithm_usage.generations == expected_gens,
                       "algorithm_usage.generations matches fevals-derived value");
    }


    {
        hpoea::wrappers::problems::SphereProblem sphere(2);
        const std::size_t feval_budget = 60u;
        const std::size_t pop = 10u;

        Budget feval_cap;
        feval_cap.function_evaluations = feval_budget;

        ParameterSet base_params;
        base_params.emplace("population_size", std::int64_t{static_cast<std::int64_t>(pop)});
        base_params.emplace("generations", std::int64_t{100});

        auto run_and_check = [&](auto &factory, ParameterSet params, const std::string &label) {
            auto algo = factory.create();
            algo->configure(params);
            auto result = algo->run(sphere, feval_cap, 42UL);
            HPOEA_V2_CHECK(runner,
                           result.algorithm_usage.function_evaluations <= feval_budget,
                           label + " fevals within budget");
            HPOEA_V2_CHECK(runner,
                           result.algorithm_usage.function_evaluations > 0u,
                           label + " fevals not zero");
            HPOEA_V2_CHECK(runner,
                           result.status == hpoea::core::RunStatus::Success,
                           label + " status is Success after generation clamp");
            HPOEA_V2_CHECK(runner,
                           std::isfinite(result.best_fitness),
                           label + " best_fitness is finite");
        };


        {
            hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
            ParameterSet params = base_params;
            run_and_check(factory, params, "DE feval cap");
        }


        {
            hpoea::pagmo_wrappers::PagmoParticleSwarmOptimizationFactory factory;
            ParameterSet params = base_params;
            run_and_check(factory, params, "PSO feval cap");
        }


        {
            hpoea::pagmo_wrappers::PagmoSelfAdaptiveDEFactory factory;
            ParameterSet params = base_params;
            run_and_check(factory, params, "SADE feval cap");
        }


        {
            hpoea::pagmo_wrappers::PagmoSgaFactory factory;
            ParameterSet params = base_params;
            run_and_check(factory, params, "SGA feval cap");
        }


        {
            hpoea::pagmo_wrappers::PagmoDe1220Factory factory;
            ParameterSet params = base_params;
            run_and_check(factory, params, "DE1220 feval cap");
        }


        {
            hpoea::pagmo_wrappers::PagmoCmaesFactory factory;
            ParameterSet params = base_params;
            run_and_check(factory, params, "CMA-ES feval cap");
        }
    }


    {
        ParameterSet params;
        params.emplace("pop", std::int64_t{42});
        auto val = hpoea::pagmo_wrappers::get_param<std::int64_t>(params, "pop");
        HPOEA_V2_CHECK(runner, val == 42u, "get_param<int64_t> returns correct value");
    }


    {
        ParameterSet params;
        bool threw = false;
        try {
            (void)hpoea::pagmo_wrappers::get_param<std::int64_t>(params, "missing");
        } catch (const std::invalid_argument &ex) {
            threw = true;

            HPOEA_V2_CHECK(runner, contains_all(ex.what(), {"missing"}),
                           "get_param missing message identifies missing parameter");
        }
        HPOEA_V2_CHECK(runner, threw, "get_param<int64_t> throws on missing param");
    }


    {
        ParameterSet params;
        params.emplace("pop", 3.14);
        bool threw = false;
        try {
            (void)hpoea::pagmo_wrappers::get_param<std::int64_t>(params, "pop");
        } catch (const std::invalid_argument &ex) {
            threw = true;
            HPOEA_V2_CHECK(runner, contains_all(ex.what(), {"pop", "type"}),
                           "get_param<int64_t> type mismatch message identifies parameter and type");
        }
        HPOEA_V2_CHECK(runner, threw, "get_param<int64_t> throws on type mismatch");
    }


    {
        ParameterSet params;
        params.emplace("pop", std::int64_t{-5});
        bool threw = false;
        try {
            (void)hpoea::pagmo_wrappers::get_param<std::int64_t>(params, "pop");
        } catch (const std::invalid_argument &ex) {
            threw = true;
            HPOEA_V2_CHECK(runner, contains_all(ex.what(), {"pop", "negative"}),
                           "get_param negative message identifies parameter and negative value");
        }
        HPOEA_V2_CHECK(runner, threw, "get_param<int64_t> throws on negative value");
    }


    {
        ParameterSet params;
        params.emplace("rate", 0.75);
        auto val = hpoea::pagmo_wrappers::get_param<double>(params, "rate");
        HPOEA_V2_CHECK(runner, val == 0.75, "get_param<double> returns correct value");
    }


    {
        ParameterSet params;
        params.emplace("rate", std::int64_t{3});
        bool threw = false;
        try {
            (void)hpoea::pagmo_wrappers::get_param<double>(params, "rate");
        } catch (const std::invalid_argument &ex) {
            threw = true;
            HPOEA_V2_CHECK(runner, contains_all(ex.what(), {"rate", "type"}),
                           "get_param<double> type mismatch message identifies parameter and type");
        }
        HPOEA_V2_CHECK(runner, threw, "get_param<double> throws on type mismatch");
    }


    {
        ParameterSet params;
        params.emplace("flag", true);
        auto val = hpoea::pagmo_wrappers::get_param<bool>(params, "flag");
        HPOEA_V2_CHECK(runner, val == true, "get_param<bool> returns correct value");
    }


    {
        ParameterSet params;
        params.emplace("flag", 1.0);
        bool threw = false;
        try {
            (void)hpoea::pagmo_wrappers::get_param<bool>(params, "flag");
        } catch (const std::invalid_argument &ex) {
            threw = true;
            HPOEA_V2_CHECK(runner, contains_all(ex.what(), {"flag", "type"}),
                           "get_param<bool> type mismatch message identifies parameter and type");
        }
        HPOEA_V2_CHECK(runner, threw, "get_param<bool> throws on type mismatch");
    }


    {
        Budget budget;
        budget.function_evaluations = 100u;
        AlgorithmRunUsage usage;
        usage.function_evaluations = 100u;
        check_algorithm_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::Success, "ok",
                                      "algorithm fevals exactly at budget");

        usage.function_evaluations = 101u;
        check_algorithm_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::BudgetExceeded, "function-evaluations budget exceeded",
                                      "algorithm fevals over budget");
    }


    {
        Budget budget;
        budget.wall_time = std::chrono::milliseconds{500};
        AlgorithmRunUsage usage;
        usage.wall_time = std::chrono::milliseconds{500};
        check_algorithm_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::Success, "ok",
                                      "algorithm wall time exactly at budget");

        usage.wall_time = std::chrono::milliseconds{501};
        check_algorithm_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::BudgetExceeded, "wall-time budget exceeded",
                                      "algorithm wall time over budget");
    }


    {
        Budget budget;
        budget.generations = 10u;
        AlgorithmRunUsage usage;
        usage.generations = 10u;
        check_algorithm_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::Success, "ok",
                                      "algorithm generations exactly at budget");

        usage.generations = 11u;
        check_algorithm_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::BudgetExceeded, "generation budget exceeded",
                                      "algorithm generations over budget");
    }


    {
        Budget budget;
        budget.function_evaluations = 1u;
        AlgorithmRunUsage usage;
        usage.function_evaluations = 999u;
        check_algorithm_budget_status(budget, usage, RunStatus::InvalidConfiguration, "bad config",
                                      RunStatus::InvalidConfiguration, "bad config",
                                      "algorithm non-success status passes through");
    }


    {
        Budget budget;
        budget.function_evaluations = 100u;
        OptimizerRunUsage usage;
        usage.objective_calls = 100u;
        check_optimizer_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::Success, "ok",
                                      "optimizer fevals exactly at budget");

        usage.objective_calls = 101u;
        check_optimizer_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::BudgetExceeded, "function-evaluations budget exceeded",
                                      "optimizer fevals over budget");
    }


    {
        Budget budget;
        budget.wall_time = std::chrono::milliseconds{10};
        OptimizerRunUsage usage;
        usage.wall_time = std::chrono::milliseconds{10};
        check_optimizer_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::Success, "ok",
                                      "optimizer wall time exactly at budget");

        usage.wall_time = std::chrono::milliseconds{11};
        check_optimizer_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::BudgetExceeded, "wall-time budget exceeded",
                                      "optimizer wall time over budget");
    }


    {
        Budget budget;
        budget.generations = 10u;
        OptimizerRunUsage usage;
        usage.iterations = 10u;
        check_optimizer_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::Success, "ok",
                                      "optimizer generations exactly at budget");

        usage.iterations = 11u;
        check_optimizer_budget_status(budget, usage, RunStatus::Success, "ok",
                                      RunStatus::BudgetExceeded, "generation budget exceeded",
                                      "optimizer generations over budget");
    }


    {
        Budget budget;
        budget.function_evaluations = 1u;
        OptimizerRunUsage usage;
        usage.objective_calls = 100u;
        check_optimizer_budget_status(budget, usage, RunStatus::FailedEvaluation, "eval failed",
                                      RunStatus::FailedEvaluation, "eval failed",
                                      "optimizer non-success status passes through");
    }


    {
        hpoea::wrappers::problems::SphereProblem problem(3);
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        auto algo = factory.create();
        ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        algo->configure(params);

        Budget budget;
        budget.function_evaluations = 10u;

        auto result = algo->run(problem, budget, 42UL);

        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::BudgetExceeded,
                       "zero-generation run should report BudgetExceeded, not Success");
        HPOEA_V2_CHECK(runner,
                       result.message.find("insufficient") != std::string::npos,
                       "zero-generations run message should mention insufficient budget");
    }

    return runner.summarize("budget_util_tests");
}
