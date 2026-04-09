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
#include <limits>
#include <string>

using hpoea::core::Budget;
using hpoea::core::AlgorithmRunUsage;
using hpoea::core::OptimizerRunUsage;
using hpoea::core::ParameterSet;
using hpoea::core::RunStatus;

int main() {
    hpoea::tests_v2::TestRunner runner;


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
        Budget budget;
        budget.function_evaluations = 10u;
        AlgorithmRunUsage usage;
        usage.function_evaluations = 11u;
        auto status = hpoea::core::RunStatus::Success;
        std::string message = "ok";
        hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::BudgetExceeded,
                       "apply_budget_status sets BudgetExceeded when fevals exceed");
        HPOEA_V2_CHECK(runner, message.find("function-evaluations") != std::string::npos,
                       "apply_budget_status sets feval message");
    }

    {
        Budget budget;
        budget.wall_time = std::chrono::milliseconds{1};
        AlgorithmRunUsage usage;
        usage.wall_time = std::chrono::milliseconds{2};
        auto status = hpoea::core::RunStatus::Success;
        std::string message;
        hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::BudgetExceeded,
                       "apply_budget_status sets BudgetExceeded when wall time exceeded");
        HPOEA_V2_CHECK(runner, message == "wall-time budget exceeded",
                       "apply_budget_status sets wall-time message");
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
                           result.status == hpoea::core::RunStatus::Success ||
                               result.status == hpoea::core::RunStatus::BudgetExceeded,
                           label + " status is Success or BudgetExceeded");
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

            HPOEA_V2_CHECK(runner, std::string(ex.what()).find("missing") != std::string::npos,
                           "get_param missing: message contains param name");
        }
        HPOEA_V2_CHECK(runner, threw, "get_param<int64_t> throws on missing param");
    }


    {
        ParameterSet params;
        params.emplace("pop", 3.14);
        bool threw = false;
        try {
            (void)hpoea::pagmo_wrappers::get_param<std::int64_t>(params, "pop");
        } catch (const std::invalid_argument &) {
            threw = true;
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
            HPOEA_V2_CHECK(runner, std::string(ex.what()).find("negative") != std::string::npos,
                           "get_param negative: message mentions negative");
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
        } catch (const std::invalid_argument &) {
            threw = true;
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
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "get_param<bool> throws on type mismatch");
    }


    {
        Budget budget;
        budget.function_evaluations = 10u;
        OptimizerRunUsage usage;
        usage.objective_calls = 11u;
        auto status = RunStatus::Success;
        std::string message;
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == RunStatus::BudgetExceeded,
                       "apply_optimizer_budget_status: fevals exceeded");
    }


    {
        Budget budget;
        budget.wall_time = std::chrono::milliseconds{10};
        OptimizerRunUsage usage;
        usage.wall_time = std::chrono::milliseconds{11};
        auto status = RunStatus::Success;
        std::string message;
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == RunStatus::BudgetExceeded,
                       "apply_optimizer_budget_status: wall_time exceeded");
    }


    {
        Budget budget;
        budget.generations = 5u;
        OptimizerRunUsage usage;
        usage.iterations = 6u;
        auto status = RunStatus::Success;
        std::string message;
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == RunStatus::BudgetExceeded,
                       "apply_optimizer_budget_status: generations exceeded");
    }


    {
        Budget budget;
        budget.function_evaluations = 100u;
        budget.generations = 10u;
        OptimizerRunUsage usage;
        usage.objective_calls = 50u;
        usage.iterations = 5u;
        auto status = RunStatus::Success;
        std::string message = "ok";
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == RunStatus::Success,
                       "apply_optimizer_budget_status: within budget remains Success");
    }


    {
        Budget budget;
        budget.function_evaluations = 1u;
        OptimizerRunUsage usage;
        usage.objective_calls = 100u;
        auto status = RunStatus::FailedEvaluation;
        std::string message = "eval failed";
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == RunStatus::FailedEvaluation,
                       "apply_optimizer_budget_status: non-Success passes through");
        HPOEA_V2_CHECK(runner, message == "eval failed",
                       "apply_optimizer_budget_status: message unchanged on passthrough");
    }


    {

        {
            Budget budget;
            budget.function_evaluations = 100u;
            AlgorithmRunUsage usage;
            usage.function_evaluations = 100u;
            auto status = RunStatus::Success;
            std::string message = "ok";
            hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
            HPOEA_V2_CHECK(runner, status == RunStatus::Success,
                           "apply_budget_status: fevals exactly at budget remains Success");
        }

        {
            Budget budget;
            budget.function_evaluations = 100u;
            AlgorithmRunUsage usage;
            usage.function_evaluations = 101u;
            auto status = RunStatus::Success;
            std::string message = "ok";
            hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
            HPOEA_V2_CHECK(runner, status == RunStatus::BudgetExceeded,
                           "apply_budget_status: fevals at budget+1 becomes BudgetExceeded");
        }

        {
            Budget budget;
            budget.wall_time = std::chrono::milliseconds{500};
            AlgorithmRunUsage usage;
            usage.wall_time = std::chrono::milliseconds{500};
            auto status = RunStatus::Success;
            std::string message = "ok";
            hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
            HPOEA_V2_CHECK(runner, status == RunStatus::Success,
                           "apply_budget_status: wall_time exactly at budget remains Success");
        }

        {
            Budget budget;
            budget.wall_time = std::chrono::milliseconds{500};
            AlgorithmRunUsage usage;
            usage.wall_time = std::chrono::milliseconds{501};
            auto status = RunStatus::Success;
            std::string message = "ok";
            hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
            HPOEA_V2_CHECK(runner, status == RunStatus::BudgetExceeded,
                           "apply_budget_status: wall_time at budget+1 becomes BudgetExceeded");
        }

        {
            Budget budget;
            budget.generations = 10u;
            AlgorithmRunUsage usage;
            usage.generations = 10u;
            auto status = RunStatus::Success;
            std::string message = "ok";
            hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
            HPOEA_V2_CHECK(runner, status == RunStatus::Success,
                           "apply_budget_status: generations exactly at budget remains Success");
        }

        {
            Budget budget;
            budget.generations = 10u;
            AlgorithmRunUsage usage;
            usage.generations = 11u;
            auto status = RunStatus::Success;
            std::string message = "ok";
            hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
            HPOEA_V2_CHECK(runner, status == RunStatus::BudgetExceeded,
                           "apply_budget_status: generations at budget+1 becomes BudgetExceeded");
        }
    }


    {
        Budget budget;
        budget.function_evaluations = 1u;
        AlgorithmRunUsage usage;
        usage.function_evaluations = 999u;
        auto status = RunStatus::InvalidConfiguration;
        std::string message = "bad config";
        hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == RunStatus::InvalidConfiguration,
                       "apply_budget_status: InvalidConfiguration passes through unchanged");
        HPOEA_V2_CHECK(runner, message == "bad config",
                       "apply_budget_status: message unchanged on non-Success passthrough");
    }


    {
        hpoea::core::ParameterSet params;
        params.emplace("x", 1.5);
        bool threw = false;
        try { (void)hpoea::pagmo_wrappers::get_param<std::int64_t>(params, "x"); }
        catch (const std::invalid_argument &) { threw = true; }
        HPOEA_V2_CHECK(runner, threw, "get_param throws on type mismatch");
    }


    {
        hpoea::core::ParameterSet params;
        bool threw = false;
        try { (void)hpoea::pagmo_wrappers::get_param<double>(params, "missing"); }
        catch (const std::invalid_argument &) { threw = true; }
        HPOEA_V2_CHECK(runner, threw, "get_param throws on missing parameter");
    }


    {
        hpoea::core::ParameterSet params;
        params.emplace("n", std::int64_t{-5});
        bool threw = false;
        try { (void)hpoea::pagmo_wrappers::get_param<std::int64_t>(params, "n"); }
        catch (const std::invalid_argument &) { threw = true; }
        HPOEA_V2_CHECK(runner, threw, "get_param rejects negative int64");
    }


    {
        hpoea::core::ParameterSet params;
        params.emplace("n", std::int64_t{42});
        auto result = hpoea::pagmo_wrappers::get_param<std::int64_t>(params, "n");
        HPOEA_V2_CHECK(runner, result == 42u, "get_param returns correct size_t for int64");
    }


    {
        hpoea::core::ParameterSet params;
        params.emplace("x", 3.14);
        auto result = hpoea::pagmo_wrappers::get_param<double>(params, "x");
        HPOEA_V2_CHECK(runner, std::abs(result - 3.14) < 1e-10, "get_param returns correct double");
    }


    {
        hpoea::core::Budget budget;
        budget.generations = 10u;
        hpoea::core::OptimizerRunUsage usage;
        usage.iterations = 10u;
        auto status = hpoea::core::RunStatus::Success;
        std::string message = "ok";
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::Success,
                       "optimizer budget: usage == limit is Success");
    }


    {
        hpoea::core::Budget budget;
        budget.generations = 10u;
        hpoea::core::OptimizerRunUsage usage;
        usage.iterations = 11u;
        auto status = hpoea::core::RunStatus::Success;
        std::string message = "ok";
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::BudgetExceeded,
                       "optimizer budget: usage > limit is BudgetExceeded");
    }


    {
        hpoea::core::Budget budget;
        budget.function_evaluations = 100u;
        hpoea::core::OptimizerRunUsage usage;
        usage.objective_calls = 100u;
        auto status = hpoea::core::RunStatus::Success;
        std::string message = "ok";
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::Success,
                       "optimizer budget: fevals == limit is Success");
    }


    {
        hpoea::core::Budget budget;
        budget.function_evaluations = 100u;
        hpoea::core::OptimizerRunUsage usage;
        usage.objective_calls = 101u;
        auto status = hpoea::core::RunStatus::Success;
        std::string message = "ok";
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::BudgetExceeded,
                       "optimizer budget: fevals > limit is BudgetExceeded");
    }


    {
        hpoea::core::Budget budget;
        budget.generations = 10u;
        hpoea::core::OptimizerRunUsage usage;
        usage.iterations = 100u;
        auto status = hpoea::core::RunStatus::InternalError;
        std::string message = "error";
        hpoea::core::apply_optimizer_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::InternalError,
                       "optimizer budget: non-Success status preserved");
    }


    {
        hpoea::core::Budget budget;
        budget.generations = 10u;
        hpoea::core::AlgorithmRunUsage usage;
        usage.generations = 10u;
        auto status = hpoea::core::RunStatus::Success;
        std::string message = "ok";
        hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::Success,
                       "budget: usage == limit is Success");
    }


    {
        hpoea::core::Budget budget;
        budget.generations = 10u;
        hpoea::core::AlgorithmRunUsage usage;
        usage.generations = 11u;
        auto status = hpoea::core::RunStatus::Success;
        std::string message = "ok";
        hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::BudgetExceeded,
                       "budget: usage > limit is BudgetExceeded");
    }


    {
        hpoea::core::Budget budget;
        budget.generations = 10u;
        hpoea::core::AlgorithmRunUsage usage;
        usage.generations = 100u;
        auto status = hpoea::core::RunStatus::FailedEvaluation;
        std::string message = "error";
        hpoea::pagmo_wrappers::apply_budget_status(budget, usage, status, message);
        HPOEA_V2_CHECK(runner, status == hpoea::core::RunStatus::FailedEvaluation,
                       "budget: FailedEvaluation preserved");
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
