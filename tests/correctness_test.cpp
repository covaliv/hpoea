// correctness tests for hpoea framework
// validates ea wrappers, hoa optimizers, parameter handling, and convergence

#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace hpoea;

namespace {

struct Test {
    std::string name;
    bool passed = false;
    std::string error;
};

std::vector<Test> tests;

// helper to run an ea on a problem and check basic validity
template <typename Factory>
bool run_ea(Factory &factory, core::IProblem &problem, std::size_t pop, std::size_t gen,
            unsigned long seed, double max_fitness) {
    auto algo = factory.create();
    core::ParameterSet params;
    params.emplace("population_size", static_cast<std::int64_t>(pop));
    params.emplace("generations", static_cast<std::int64_t>(gen));
    algo->configure(params);

    core::Budget budget;
    budget.generations = gen;

    auto result = algo->run(problem, budget, seed);
    return result.status == core::RunStatus::Success &&
           result.best_fitness < max_fitness &&
           result.best_solution.size() == problem.dimension() &&
           result.budget_usage.generations <= gen;
}

void log(const Test &t) {
    std::cout << "  " << (t.passed ? "pass" : "FAIL") << " " << t.name;
    if (!t.passed && !t.error.empty()) std::cout << " (" << t.error << ")";
    std::cout << "\n";
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "correctness tests\n\n";

    // test ea wrappers on sphere
    std::cout << "ea wrapper basic functionality\n";
    {
        wrappers::problems::SphereProblem sphere(5);

        pagmo_wrappers::PagmoDifferentialEvolutionFactory de;
        Test t{"de on sphere 5d", run_ea(de, sphere, 30, 50, 42UL, 1.0), ""};
        tests.push_back(t);
        log(t);

        pagmo_wrappers::PagmoParticleSwarmOptimizationFactory pso;
        t = {"pso on sphere 5d", run_ea(pso, sphere, 30, 50, 42UL, 1.0), ""};
        tests.push_back(t);
        log(t);

        pagmo_wrappers::PagmoSelfAdaptiveDEFactory sade;
        t = {"sade on sphere 5d", run_ea(sade, sphere, 30, 50, 42UL, 1.0), ""};
        tests.push_back(t);
        log(t);
    }

    // test de on various benchmark problems
    std::cout << "\nproblem variety\n";
    {
        pagmo_wrappers::PagmoDifferentialEvolutionFactory de;

        auto test_problem = [&](const std::string &name, core::IProblem &p) {
            Test t{name, run_ea(de, p, 50, 100, 123UL, 100.0), ""};
            tests.push_back(t);
            log(t);
        };

        wrappers::problems::SphereProblem sphere(10);
        wrappers::problems::RosenbrockProblem rosen(6);
        wrappers::problems::RastriginProblem rast(8);
        wrappers::problems::AckleyProblem ackley(5);

        test_problem("de on sphere 10d", sphere);
        test_problem("de on rosenbrock 6d", rosen);
        test_problem("de on rastrigin 8d", rast);
        test_problem("de on ackley 5d", ackley);
    }

    // reproducibility: same seed should give identical results
    std::cout << "\nreproducibility\n";
    {
        wrappers::problems::SphereProblem sphere(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory de;

        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(20));
        params.emplace("generations", static_cast<std::int64_t>(30));

        core::Budget budget;
        budget.generations = 30;

        auto run = [&]() {
            auto algo = de.create();
            algo->configure(params);
            return algo->run(sphere, budget, 999UL);
        };

        auto r1 = run();
        auto r2 = run();
        double diff = std::abs(r1.best_fitness - r2.best_fitness);

        Test t{"same seed gives identical results", diff < 1e-10, "diff=" + std::to_string(diff)};
        tests.push_back(t);
        log(t);
    }

    // budget enforcement: algorithm should respect generation limit
    std::cout << "\nbudget enforcement\n";
    {
        wrappers::problems::SphereProblem sphere(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory de;

        auto algo = de.create();
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(20));
        params.emplace("generations", static_cast<std::int64_t>(1000)); // request more than budget allows
        algo->configure(params);

        core::Budget budget;
        budget.generations = 50; // but limit to 50

        auto result = algo->run(sphere, budget, 42UL);

        Test t{"respects generation budget", result.budget_usage.generations <= 50,
               "used " + std::to_string(result.budget_usage.generations)};
        tests.push_back(t);
        log(t);
    }

    // hoa functionality: hyperparameter optimizers should improve fitness
    std::cout << "\nhoa basic functionality\n";
    {
        wrappers::problems::SphereProblem sphere(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory de;

        core::Budget budget;
        budget.generations = 10;
        budget.function_evaluations = 2000;

        auto test_hoa = [&](const std::string &name, core::IHyperparameterOptimizer &hoa) {
            core::ParameterSet hoa_params;
            hoa_params.emplace("generations", static_cast<std::int64_t>(10));
            hoa.configure(hoa_params);

            auto result = hoa.optimize(de, sphere, budget, 42UL);
            bool ok = result.status == core::RunStatus::Success &&
                      !result.trials.empty() && result.best_objective < 10.0;

            Test t{name, ok, ""};
            tests.push_back(t);
            log(t);
        };

        pagmo_wrappers::PagmoCmaesHyperOptimizer cmaes_hoa;
        pagmo_wrappers::PagmoPsoHyperOptimizer pso_hoa;

        test_hoa("cmaes hoa", cmaes_hoa);
        test_hoa("pso hoa", pso_hoa);
    }

    // parameter validation: should reject invalid parameters
    std::cout << "\nparameter validation\n";
    {
        pagmo_wrappers::PagmoDifferentialEvolutionFactory de;
        auto algo = de.create();

        core::ParameterSet invalid;
        invalid.emplace("variant", static_cast<std::int64_t>(0)); // out of range

        bool threw = false;
        try {
            algo->configure(invalid);
        } catch (const core::ParameterValidationError &) {
            threw = true;
        }

        Test t{"rejects invalid variant", threw, ""};
        tests.push_back(t);
        log(t);
    }

    // convergence: more generations should yield better fitness
    std::cout << "\nconvergence\n";
    {
        wrappers::problems::SphereProblem sphere(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory de;

        auto run_with_gen = [&](std::size_t gen) {
            auto algo = de.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(30));
            params.emplace("generations", static_cast<std::int64_t>(gen));
            algo->configure(params);

            core::Budget budget;
            budget.generations = gen;
            return algo->run(sphere, budget, 42UL).best_fitness;
        };

        double f20 = run_with_gen(20);
        double f50 = run_with_gen(50);
        double f100 = run_with_gen(100);

        // fitness should improve (decrease) with more generations
        bool ok = f20 >= f50 && f50 >= f100 && f100 < f20;

        Test t{"more generations improves fitness", ok,
               "f20=" + std::to_string(f20) + " f50=" + std::to_string(f50) + " f100=" + std::to_string(f100)};
        tests.push_back(t);
        log(t);
    }

    // summary
    std::cout << "\nsummary\n";
    std::size_t passed = std::count_if(tests.begin(), tests.end(), [](const Test &t) { return t.passed; });
    std::size_t failed = tests.size() - passed;

    std::cout << "  total: " << tests.size() << ", passed: " << passed << ", failed: " << failed << "\n";

    if (failed > 0) {
        std::cout << "\nfailed tests:\n";
        for (const auto &t : tests) {
            if (!t.passed) std::cout << "  - " << t.name << ": " << t.error << "\n";
        }
        return 1;
    }

    std::cout << "\nall tests passed\n";
    return 0;
}

