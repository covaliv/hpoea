#include "test_harness.hpp"
#include "test_fixtures.hpp"

#include "hpoea/wrappers/pagmo/cmaes_algorithm.hpp"
#include "hpoea/wrappers/pagmo/de1220_algorithm.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sga_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <limits>
#include <string>

namespace {

using hpoea::tests_v2::ThrowingProblem;

template <typename Factory>
hpoea::core::OptimizationResult run_algo(Factory &factory,
                                         const hpoea::core::IProblem &problem,
                                         const hpoea::core::ParameterSet &params,
                                         const hpoea::core::Budget &budget,
                                         unsigned long seed) {
    auto algo = factory.create();
    algo->configure(params);
    return algo->run(problem, budget, seed);
}

bool vector_equal(const std::vector<double> &lhs, const std::vector<double> &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    hpoea::wrappers::problems::SphereProblem sphere(4);
    hpoea::core::Budget budget;
    budget.generations = 5u;


    {
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        params.emplace("scaling_factor", 0.7);
        params.emplace("crossover_rate", 0.9);
        params.emplace("variant", std::int64_t{2});

        const auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "DE run returns Success");
        HPOEA_V2_CHECK(runner, result.best_solution.size() == sphere.dimension(),
                       "DE best_solution has correct dimension");
        HPOEA_V2_CHECK(runner, std::isfinite(result.best_fitness),
                       "DE best_fitness is finite");

        const auto result2 = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.best_fitness == result2.best_fitness,
                       "DE is deterministic for same seed");
        HPOEA_V2_CHECK(runner, vector_equal(result.best_solution, result2.best_solution),
                       "DE best_solution identical for same seed");

        HPOEA_V2_CHECK(runner, result.algorithm_usage.generations <= 5u,
                       "DE generations within budget");
        HPOEA_V2_CHECK(runner, result.algorithm_usage.generations > 0u,
                       "DE generations positive when budget allows");

        hpoea::core::Budget zero_budget;
        zero_budget.generations = 0u;
        const auto zero_result = run_algo(factory, sphere, params, zero_budget, 7UL);
        HPOEA_V2_CHECK(runner, zero_result.algorithm_usage.generations == 0u,
                       "DE respects zero generation budget");

        bool threw = false;
        try {
            hpoea::core::ParameterSet bad;
            bad.emplace("population_size", std::int64_t{1});
            auto algo = factory.create();
            algo->configure(bad);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "DE rejects invalid population size");
    }


    {
        hpoea::pagmo_wrappers::PagmoParticleSwarmOptimizationFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        params.emplace("omega", 0.7);
        params.emplace("eta1", 2.0);
        params.emplace("eta2", 2.0);
        params.emplace("max_velocity", 0.5);
        params.emplace("variant", std::int64_t{5});

        const auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "PSO run returns Success");
        HPOEA_V2_CHECK(runner, result.best_solution.size() == sphere.dimension(),
                       "PSO best_solution has correct dimension");
        HPOEA_V2_CHECK(runner, std::isfinite(result.best_fitness),
                       "PSO best_fitness is finite");

        const auto result2 = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.best_fitness == result2.best_fitness,
                       "PSO deterministic for same seed");
        HPOEA_V2_CHECK(runner, vector_equal(result.best_solution, result2.best_solution),
                       "PSO best_solution identical for same seed");

        bool threw = false;
        try {
            hpoea::core::ParameterSet bad;
            bad.emplace("population_size", std::int64_t{1});
            auto algo = factory.create();
            algo->configure(bad);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "PSO rejects invalid population size");
    }


    {
        hpoea::pagmo_wrappers::PagmoSelfAdaptiveDEFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        params.emplace("variant", std::int64_t{2});
        params.emplace("variant_adptv", std::int64_t{1});

        const auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "SADE run returns Success");
        HPOEA_V2_CHECK(runner, result.best_solution.size() == sphere.dimension(),
                       "SADE best_solution has correct dimension");
        HPOEA_V2_CHECK(runner, std::isfinite(result.best_fitness),
                       "SADE best_fitness is finite");

        const auto result2 = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.best_fitness == result2.best_fitness,
                       "SADE is deterministic for same seed");
        HPOEA_V2_CHECK(runner, vector_equal(result.best_solution, result2.best_solution),
                       "SADE best_solution identical for same seed");

        bool threw = false;
        try {
            hpoea::core::ParameterSet bad;
            bad.emplace("population_size", std::int64_t{1});
            auto algo = factory.create();
            algo->configure(bad);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "SADE rejects invalid population size");
    }


    {
        hpoea::pagmo_wrappers::PagmoSgaFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        params.emplace("crossover_probability", 0.9);
        params.emplace("mutation_probability", 0.02);

        const auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "SGA run returns Success");
        HPOEA_V2_CHECK(runner, result.best_solution.size() == sphere.dimension(),
                       "SGA best_solution has correct dimension");
        HPOEA_V2_CHECK(runner, std::isfinite(result.best_fitness),
                       "SGA best_fitness is finite");

        const auto result2 = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.best_fitness == result2.best_fitness,
                       "SGA is deterministic for same seed");
        HPOEA_V2_CHECK(runner, vector_equal(result.best_solution, result2.best_solution),
                       "SGA best_solution identical for same seed");

        bool threw = false;
        try {
            hpoea::core::ParameterSet bad;
            bad.emplace("population_size", std::int64_t{1});
            auto algo = factory.create();
            algo->configure(bad);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "SGA rejects invalid population size");
    }


    {
        hpoea::pagmo_wrappers::PagmoDe1220Factory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        params.emplace("variant_adaptation", std::int64_t{1});
        params.emplace("memory", false);

        const auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "DE1220 run returns Success");
        HPOEA_V2_CHECK(runner, result.best_solution.size() == sphere.dimension(),
                       "DE1220 best_solution has correct dimension");
        HPOEA_V2_CHECK(runner, std::isfinite(result.best_fitness),
                       "DE1220 best_fitness is finite");

        const auto result2 = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.best_fitness == result2.best_fitness,
                       "DE1220 is deterministic for same seed");
        HPOEA_V2_CHECK(runner, vector_equal(result.best_solution, result2.best_solution),
                       "DE1220 best_solution identical for same seed");

        bool threw = false;
        try {
            hpoea::core::ParameterSet bad;
            bad.emplace("population_size", std::int64_t{1});
            auto algo = factory.create();
            algo->configure(bad);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "DE1220 rejects invalid population size");
    }


    {
        hpoea::pagmo_wrappers::PagmoCmaesFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        params.emplace("sigma0", 0.5);
        params.emplace("ftol", 1e-6);
        params.emplace("xtol", 1e-6);

        const auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "CMA-ES run returns Success");
        HPOEA_V2_CHECK(runner, result.best_solution.size() == sphere.dimension(),
                       "CMA-ES best_solution has correct dimension");
        HPOEA_V2_CHECK(runner, std::isfinite(result.best_fitness),
                       "CMA-ES best_fitness is finite");

        const auto result2 = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.best_fitness == result2.best_fitness,
                       "CMA-ES is deterministic for same seed");
        HPOEA_V2_CHECK(runner, vector_equal(result.best_solution, result2.best_solution),
                       "CMA-ES best_solution identical for same seed");

        bool threw = false;
        try {
            hpoea::core::ParameterSet bad;
            bad.emplace("population_size", std::int64_t{1});
            auto algo = factory.create();
            algo->configure(bad);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "CMA-ES rejects invalid population size");
    }


    {
        ThrowingProblem throwing_problem;
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{2});
        hpoea::core::Budget local_budget;
        local_budget.generations = 2u;
        const auto result = run_algo(factory, throwing_problem, params, local_budget, 99UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::FailedEvaluation,
                       "evaluation exceptions map to FailedEvaluation");
        HPOEA_V2_CHECK(runner, result.error_info.has_value() &&
                                  result.error_info->category == "evaluation_failure",
                       "evaluation failures include error_info category");
    }


    {
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{10});
        params.emplace("scaling_factor", 0.7);
        params.emplace("crossover_rate", 0.9);
        params.emplace("variant", std::int64_t{2});

        hpoea::core::Budget local_budget;
        local_budget.generations = 10u;

        const auto r1 = run_algo(factory, sphere, params, local_budget, 42UL);
        const auto r2 = run_algo(factory, sphere, params, local_budget, 99UL);
        HPOEA_V2_CHECK(runner, r1.best_fitness != r2.best_fitness,
                       "DE different seeds produce different fitness");
    }


    {
        ThrowingProblem throwing_problem;
        hpoea::pagmo_wrappers::PagmoParticleSwarmOptimizationFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{2});
        hpoea::core::Budget local_budget;
        local_budget.generations = 2u;
        const auto result = run_algo(factory, throwing_problem, params, local_budget, 99UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::FailedEvaluation,
                       "PSO: evaluation exceptions map to FailedEvaluation");
    }


    {
        ThrowingProblem throwing_problem;
        hpoea::pagmo_wrappers::PagmoSelfAdaptiveDEFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{2});
        hpoea::core::Budget local_budget;
        local_budget.generations = 2u;
        const auto result = run_algo(factory, throwing_problem, params, local_budget, 99UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::FailedEvaluation,
                       "SADE: evaluation exceptions map to FailedEvaluation");
    }


    {
        ThrowingProblem throwing_problem;
        hpoea::pagmo_wrappers::PagmoSgaFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{2});
        hpoea::core::Budget local_budget;
        local_budget.generations = 2u;
        const auto result = run_algo(factory, throwing_problem, params, local_budget, 99UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::FailedEvaluation,
                       "SGA: evaluation exceptions map to FailedEvaluation");
    }


    {
        ThrowingProblem throwing_problem;
        hpoea::pagmo_wrappers::PagmoDe1220Factory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{2});
        hpoea::core::Budget local_budget;
        local_budget.generations = 2u;
        const auto result = run_algo(factory, throwing_problem, params, local_budget, 99UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::FailedEvaluation,
                       "DE1220: evaluation exceptions map to FailedEvaluation");
    }


    {
        ThrowingProblem throwing_problem;
        hpoea::pagmo_wrappers::PagmoCmaesFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{2});
        hpoea::core::Budget local_budget;
        local_budget.generations = 2u;
        const auto result = run_algo(factory, throwing_problem, params, local_budget, 99UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::FailedEvaluation,
                       "CMA-ES: evaluation exceptions map to FailedEvaluation");
    }


    {
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        params.emplace("scaling_factor", 0.7);
        params.emplace("crossover_rate", 0.9);
        params.emplace("variant", std::int64_t{2});

        const auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.algorithm_usage.function_evaluations > 0u,
                       "DE function_evaluations is positive");
        HPOEA_V2_CHECK(runner, result.algorithm_usage.function_evaluations >= 20u,
                       "DE function_evaluations at least population_size");
    }


    {
        auto algo = hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory().create();
        hpoea::core::ParameterSet bad_params;
        bad_params["crossover_rate"] = 5.0;
        bool threw = false;
        try {
            algo->configure(bad_params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw,
            "configure() with out-of-range param throws ParameterValidationError");
    }


    {
        hpoea::wrappers::problems::SphereProblem sphere(2);
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        hpoea::core::Budget budget;
        budget.generations = 10;

        for (auto variant : {std::int64_t{1}, std::int64_t{10}}) {
            hpoea::core::ParameterSet params;
            params.emplace("population_size", std::int64_t{20});
            params.emplace("generations", std::int64_t{10});
            params.emplace("crossover_rate", 0.9);
            params.emplace("scaling_factor", 0.8);
            params.emplace("variant", variant);
            params.emplace("ftol", 1e-12);
            params.emplace("xtol", 1e-12);

            auto result = run_algo(factory, sphere, params, budget, 42UL);
            HPOEA_V2_CHECK(runner,
                result.status == hpoea::core::RunStatus::Success,
                "DE variant=" + std::to_string(variant) + " runs successfully");
            HPOEA_V2_CHECK(runner,
                std::isfinite(result.best_fitness),
                "DE variant=" + std::to_string(variant) + " produces finite fitness");
        }
    }


    {
        hpoea::wrappers::problems::SphereProblem sphere(2);
        hpoea::pagmo_wrappers::PagmoParticleSwarmOptimizationFactory factory;
        hpoea::core::Budget budget;
        budget.generations = 10;

        for (auto variant : {std::int64_t{1}, std::int64_t{6}}) {
            hpoea::core::ParameterSet params;
            params.emplace("population_size", std::int64_t{20});
            params.emplace("generations", std::int64_t{10});
            params.emplace("omega", 0.7298);
            params.emplace("eta1", 2.05);
            params.emplace("eta2", 2.05);
            params.emplace("max_velocity", 0.5);
            params.emplace("variant", variant);

            auto result = run_algo(factory, sphere, params, budget, 42UL);
            HPOEA_V2_CHECK(runner,
                result.status == hpoea::core::RunStatus::Success,
                "PSO variant=" + std::to_string(variant) + " runs successfully");
            HPOEA_V2_CHECK(runner,
                std::isfinite(result.best_fitness),
                "PSO variant=" + std::to_string(variant) + " produces finite fitness");
        }
    }


    {
        hpoea::wrappers::problems::SphereProblem sphere(2);
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        hpoea::core::Budget budget;
        budget.generations = 10;


        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{10});
        params.emplace("crossover_rate", 1.0);
        params.emplace("scaling_factor", 0.0);
        params.emplace("variant", std::int64_t{2});
        params.emplace("ftol", 1e-12);
        params.emplace("xtol", 1e-12);

        auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner,
            result.status == hpoea::core::RunStatus::Success,
            "DE scaling_factor=0 crossover_rate=1 runs successfully");
        HPOEA_V2_CHECK(runner,
            std::isfinite(result.best_fitness),
            "DE boundary params produce finite fitness");
    }

    return runner.summarize("evolutionary_algorithms_tests");
}
