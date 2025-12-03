#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

using namespace hpoea;

namespace {

const bool verbose = [] {
    if (const char *v = std::getenv("HPOEA_LOG_RESULTS")) {
        return std::string_view{v} == "1";
    }
    return false;
}();

const std::vector<unsigned long> seeds{42UL, 1337UL, 2024UL};

struct TestCase {
    std::string name;
    std::unique_ptr<core::IProblem> problem;
    core::ParameterSet params;
    core::Budget budget;
    double max_fitness;
};

// run test across seeds, return true if all pass
template <typename Factory>
bool run(Factory &factory, const TestCase &tc) {
    double worst = 0.0;

    for (auto seed : seeds) {
        auto algo = factory.create();
        algo->configure(tc.params);
        auto result = algo->run(*tc.problem, tc.budget, seed);

        if (result.status != core::RunStatus::Success) {
            std::cerr << tc.name << " seed=" << seed << " failed: " << result.message << "\n";
            return false;
        }

        if (result.best_solution.size() != tc.problem->dimension()) {
            std::cerr << tc.name << " seed=" << seed << " invalid solution size\n";
            return false;
        }

        if (tc.budget.generations && result.budget_usage.generations > *tc.budget.generations) {
            std::cerr << tc.name << " seed=" << seed << " exceeded budget\n";
            return false;
        }

        worst = std::max(worst, result.best_fitness);

        if (verbose) {
            std::cout << std::fixed << std::setprecision(6)
                      << tc.name << " seed=" << seed
                      << " fitness=" << result.best_fitness << "\n";
        }
    }

    if (worst > tc.max_fitness) {
        std::cerr << tc.name << " worst=" << worst << " exceeds limit=" << tc.max_fitness << "\n";
        return false;
    }

    return true;
}

// test parameter validation throws on invalid input
template <typename Factory>
bool test_validation(Factory &factory, const std::string &param, auto invalid_value) {
    auto algo = factory.create();
    core::ParameterSet params;
    params.emplace(param, invalid_value);

    try {
        algo->configure(params);
        std::cerr << "expected ParameterValidationError for " << param << "\n";
        return false;
    } catch (const core::ParameterValidationError &) {
        return true;
    }
}

} // namespace

int main() {
    int failures = 0;

    // de tests
    std::cout << "de wrapper\n";
    {
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;

        TestCase tc;
        tc.name = "  sphere";
        tc.problem = std::make_unique<wrappers::problems::SphereProblem>(10);
        tc.params.emplace("population_size", static_cast<std::int64_t>(100));
        tc.params.emplace("generations", static_cast<std::int64_t>(200));
        tc.params.emplace("scaling_factor", 0.7);
        tc.params.emplace("crossover_rate", 0.9);
        tc.budget.generations = 200;
        tc.max_fitness = 0.1;

        if (!run(factory, tc)) failures++;

        tc.name = "  rosenbrock";
        tc.problem = std::make_unique<wrappers::problems::RosenbrockProblem>(6);
        tc.params.clear();
        tc.params.emplace("population_size", static_cast<std::int64_t>(100));
        tc.params.emplace("generations", static_cast<std::int64_t>(300));
        tc.budget.generations = 300;
        tc.max_fitness = 1.0;

        if (!run(factory, tc)) failures++;

        if (!test_validation(factory, "variant", static_cast<std::int64_t>(0))) failures++;
    }

    // pso tests
    std::cout << "pso wrapper\n";
    {
        pagmo_wrappers::PagmoParticleSwarmOptimizationFactory factory;

        TestCase tc;
        tc.name = "  sphere";
        tc.problem = std::make_unique<wrappers::problems::SphereProblem>(10);
        tc.params.emplace("population_size", static_cast<std::int64_t>(50));
        tc.params.emplace("generations", static_cast<std::int64_t>(200));
        tc.params.emplace("omega", 0.7298);
        tc.params.emplace("eta1", 2.05);
        tc.params.emplace("eta2", 2.05);
        tc.budget.generations = 200;
        tc.max_fitness = 0.1;

        if (!run(factory, tc)) failures++;

        tc.name = "  rastrigin";
        tc.problem = std::make_unique<wrappers::problems::RastriginProblem>(6);
        tc.params.clear();
        tc.params.emplace("population_size", static_cast<std::int64_t>(60));
        tc.params.emplace("generations", static_cast<std::int64_t>(300));
        tc.budget.generations = 300;
        tc.max_fitness = 10.0;

        if (!run(factory, tc)) failures++;

        if (!test_validation(factory, "omega", 2.0)) failures++;
    }

    // sade tests
    std::cout << "sade wrapper\n";
    {
        pagmo_wrappers::PagmoSelfAdaptiveDEFactory factory;

        TestCase tc;
        tc.name = "  sphere";
        tc.problem = std::make_unique<wrappers::problems::SphereProblem>(10);
        tc.params.emplace("population_size", static_cast<std::int64_t>(50));
        tc.params.emplace("generations", static_cast<std::int64_t>(200));
        tc.budget.generations = 200;
        tc.max_fitness = 0.1;

        if (!run(factory, tc)) failures++;

        tc.name = "  rosenbrock";
        tc.problem = std::make_unique<wrappers::problems::RosenbrockProblem>(6);
        tc.params.clear();
        tc.params.emplace("population_size", static_cast<std::int64_t>(60));
        tc.params.emplace("generations", static_cast<std::int64_t>(300));
        tc.budget.generations = 300;
        tc.max_fitness = 5.0;

        if (!run(factory, tc)) failures++;

        if (!test_validation(factory, "variant", static_cast<std::int64_t>(0))) failures++;
    }

    if (failures > 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << "all wrapper tests passed\n";
    return 0;
}
