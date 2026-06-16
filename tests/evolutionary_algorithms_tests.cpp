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
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

hpoea::core::ParameterSet endpoint_params(const hpoea::core::ParameterSpace &space, const char *endpoint) {
    const bool use_max = std::string_view{endpoint} == "max";
    hpoea::core::ParameterSet params;
    for (const auto &desc : space.descriptors()) {
        if (desc.type == hpoea::core::ParameterType::Continuous) {
            const auto &range = *desc.continuous_range;
            params.emplace(desc.name, use_max ? range.upper : range.lower);
        } else if (desc.type == hpoea::core::ParameterType::Integer) {
            const auto &range = *desc.integer_range;
            params.emplace(desc.name, use_max ? range.upper : range.lower);
        } else if (desc.type == hpoea::core::ParameterType::Boolean) {
            params.emplace(desc.name, false);
        }
    }
    return params;
}

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    hpoea::wrappers::problems::SphereProblem sphere(4);
    hpoea::core::Budget budget;
    budget.generations = 5u;


    {
        // smoke every EA wrapper
        // runs to Success and records its seed
        // reproducible
        // rejects pop=1
        // maps evaluation exceptions to FailedEvaluation
        // all six share run_population's catch
        struct AlgoCase {
            const char *name;
            std::function<std::unique_ptr<hpoea::core::IEvolutionaryAlgorithmFactory>()> make;
            std::function<void(hpoea::core::ParameterSet &)> extra;
        };
        const AlgoCase cases[] = {
            {"DE", [] { return std::make_unique<hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory>(); },
             [](hpoea::core::ParameterSet &p) {
                 p.emplace("scaling_factor", 0.7);
                 p.emplace("crossover_rate", 0.9);
                 p.emplace("variant", std::int64_t{2});
             }},
            {"PSO", [] { return std::make_unique<hpoea::pagmo_wrappers::PagmoParticleSwarmOptimizationFactory>(); },
             [](hpoea::core::ParameterSet &p) {
                 p.emplace("omega", 0.7);
                 p.emplace("eta1", 2.0);
                 p.emplace("eta2", 2.0);
                 p.emplace("max_velocity", 0.5);
                 p.emplace("variant", std::int64_t{5});
             }},
            {"SADE", [] { return std::make_unique<hpoea::pagmo_wrappers::PagmoSelfAdaptiveDEFactory>(); },
             [](hpoea::core::ParameterSet &p) {
                 p.emplace("variant", std::int64_t{2});
                 p.emplace("variant_adptv", std::int64_t{1});
             }},
            {"SGA", [] { return std::make_unique<hpoea::pagmo_wrappers::PagmoSgaFactory>(); },
             [](hpoea::core::ParameterSet &p) {
                 p.emplace("crossover_probability", 0.9);
                 p.emplace("mutation_probability", 0.02);
             }},
            {"DE1220", [] { return std::make_unique<hpoea::pagmo_wrappers::PagmoDe1220Factory>(); },
             [](hpoea::core::ParameterSet &p) {
                 p.emplace("variant_adaptation", std::int64_t{1});
                 p.emplace("memory", false);
             }},
            {"CMA-ES", [] { return std::make_unique<hpoea::pagmo_wrappers::PagmoCmaesFactory>(); },
             [](hpoea::core::ParameterSet &p) {
                 p.emplace("sigma0", 0.5);
                 p.emplace("ftol", 1e-6);
                 p.emplace("xtol", 1e-6);
             }},
        };

        for (const auto &c : cases) {
            const std::string name = c.name;
            hpoea::core::ParameterSet params;
            params.emplace("population_size", std::int64_t{20});
            params.emplace("generations", std::int64_t{5});
            c.extra(params);

            auto factory = c.make();
            const auto r1 = run_algo(*factory, sphere, params, budget, 42UL);
            HPOEA_V2_CHECK(runner, r1.status == hpoea::core::RunStatus::Success,
                           name + " run returns Success");
            HPOEA_V2_CHECK(runner, r1.best_solution.size() == sphere.dimension() &&
                                      std::isfinite(r1.best_fitness),
                           name + " produces finite best_fitness of correct dimension");
            HPOEA_V2_CHECK(runner, r1.seed == 42UL, name + " records the requested seed");

            const auto r2 = run_algo(*c.make(), sphere, params, budget, 42UL);
            HPOEA_V2_CHECK(runner, r1.best_fitness == r2.best_fitness &&
                                      vector_equal(r1.best_solution, r2.best_solution),
                           name + " deterministic for same seed");

            ThrowingProblem throwing_problem;
            hpoea::core::Budget throw_budget;
            throw_budget.generations = 2u;
            hpoea::core::ParameterSet throw_params;
            throw_params.emplace("population_size", std::int64_t{20});
            throw_params.emplace("generations", std::int64_t{2});
            const auto rt = run_algo(*c.make(), throwing_problem, throw_params, throw_budget, 99UL);
            HPOEA_V2_CHECK(runner, rt.status == hpoea::core::RunStatus::FailedEvaluation,
                           name + " maps evaluation exceptions to FailedEvaluation");

            bool threw = false;
            try {
                hpoea::core::ParameterSet bad;
                bad.emplace("population_size", std::int64_t{1});
                c.make()->create()->configure(bad);
            } catch (const hpoea::core::ParameterValidationError &) {
                threw = true;
            }
            HPOEA_V2_CHECK(runner, threw, name + " rejects invalid population size");
        }
    }


    {
        // DE respects the generation budget
        // zero-generation budget is BudgetExceeded (C06)
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{5});
        params.emplace("scaling_factor", 0.7);
        params.emplace("crossover_rate", 0.9);
        params.emplace("variant", std::int64_t{2});

        const auto result = run_algo(factory, sphere, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.algorithm_usage.generations <= 5u,
                       "DE generations within budget");
        HPOEA_V2_CHECK(runner, result.algorithm_usage.generations > 0u,
                       "DE generations positive when budget allows");

        hpoea::core::Budget zero_budget;
        zero_budget.generations = 0u;
        const auto zero_result = run_algo(factory, sphere, params, zero_budget, 7UL);
        HPOEA_V2_CHECK(runner, zero_result.status == hpoea::core::RunStatus::BudgetExceeded,
                       "DE zero generation budget returns BudgetExceeded");
        HPOEA_V2_CHECK(runner, zero_result.algorithm_usage.generations == 0u,
                       "DE respects zero generation budget");
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
        hpoea::pagmo_wrappers::PagmoParticleSwarmOptimizationFactory factory;
        const auto &space = factory.parameter_space();
        const auto &mv = space.descriptor("max_velocity");
        HPOEA_V2_CHECK(runner, mv.continuous_range.has_value(), "PSO max_velocity has continuous range");
        HPOEA_V2_CHECK(runner, mv.continuous_range->lower == 0.01,
                       "PSO max_velocity lower bound is 0.01 (pagmo requires > 0)");
        HPOEA_V2_CHECK(runner, mv.continuous_range->upper == 1.0,
                       "PSO max_velocity upper bound is 1.0 (pagmo requires <= 1)");

        bool threw = false;
        try {
            hpoea::core::ParameterSet bad;
            bad.emplace("population_size", std::int64_t{20});
            bad.emplace("max_velocity", 0.0);
            auto algo = factory.create();
            algo->configure(bad);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "PSO rejects max_velocity=0 (out of pagmo range)");
    }


    {
        struct UnboundedDescent final : hpoea::core::IProblem {
            hpoea::core::ProblemMetadata metadata_{};
            [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }
            [[nodiscard]] std::size_t dimension() const override { return 1; }
            [[nodiscard]] std::vector<double> lower_bounds() const override { return {-1.0}; }
            [[nodiscard]] std::vector<double> upper_bounds() const override { return {1.0}; }
            [[nodiscard]] double evaluate(const std::vector<double> &x) const override { return -x[0]; }
        } problem;

        hpoea::pagmo_wrappers::PagmoCmaesFactory factory;
        hpoea::core::ParameterSet params;
        params.emplace("population_size", std::int64_t{20});
        params.emplace("generations", std::int64_t{50});
        params.emplace("sigma0", 0.5);
        params.emplace("ftol", 1e-12);
        params.emplace("xtol", 1e-12);
        hpoea::core::Budget budget;
        budget.generations = 50u;

        const auto result = run_algo(factory, problem, params, budget, 42UL);
        HPOEA_V2_CHECK(runner, result.status == hpoea::core::RunStatus::Success,
                       "CMA-ES run returns Success on unbounded-descent problem");
        HPOEA_V2_CHECK(runner, result.best_solution.size() == 1u,
                       "CMA-ES best_solution has correct dimension");
        bool inside = !result.best_solution.empty() &&
                      result.best_solution[0] >= -1.0 &&
                      result.best_solution[0] <= 1.0;
        HPOEA_V2_CHECK(runner, inside,
                       "CMA-ES force_bounds keeps champion inside box bounds");
    }


    {
        hpoea::wrappers::problems::SphereProblem sphere(2);
        hpoea::core::Budget budget;
        budget.generations = 1u;

        struct Case {
            const char *name;
            std::function<std::unique_ptr<hpoea::core::IEvolutionaryAlgorithmFactory>()> make;
        };
        Case cases[] = {
            {"DE",    []{ return std::make_unique<hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory>(); }},
            {"SADE",  []{ return std::make_unique<hpoea::pagmo_wrappers::PagmoSelfAdaptiveDEFactory>(); }},
            {"DE1220",[]{ return std::make_unique<hpoea::pagmo_wrappers::PagmoDe1220Factory>(); }},
            {"PSO",   []{ return std::make_unique<hpoea::pagmo_wrappers::PagmoParticleSwarmOptimizationFactory>(); }},
            {"SGA",   []{ return std::make_unique<hpoea::pagmo_wrappers::PagmoSgaFactory>(); }},
            {"CMAES", []{ return std::make_unique<hpoea::pagmo_wrappers::PagmoCmaesFactory>(); }},
        };

        for (const auto &c : cases) {
            for (const char *endpoint : {"min", "max"}) {
                auto factory = c.make();
                auto params = endpoint_params(factory->parameter_space(), endpoint);
                params["generations"] = std::int64_t{1};
                auto algo = factory->create();
                algo->configure(params);
                bool threw = false;
                std::string msg;
                try {
                    auto result = algo->run(sphere, budget, 123UL);
                    if (result.status != hpoea::core::RunStatus::Success &&
                        result.status != hpoea::core::RunStatus::BudgetExceeded) {
                        threw = true;
                        msg = result.message;
                    }
                } catch (const std::exception &ex) {
                    threw = true;
                    msg = ex.what();
                }
                HPOEA_V2_CHECK(runner, !threw,
                    std::string(c.name) + " " + endpoint + " endpoint sweep runs without throwing" +
                    (msg.empty() ? "" : (" (msg=" + msg + ")")));
            }
        }
    }


    return runner.summarize("evolutionary_algorithms_tests");
}
