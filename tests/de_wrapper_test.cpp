#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
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

const std::vector<unsigned long> seeds{42UL, 1337UL, 2024UL, 9001UL, 123456UL};

struct TestCase {
    std::string name;
    std::unique_ptr<core::IProblem> problem;
    core::ParameterSet params;
    core::Budget budget;
    double max_fitness; // worst acceptable fitness across all seeds
};

// run test case across all seeds and return true if all pass
bool run(pagmo_wrappers::PagmoDifferentialEvolutionFactory &factory, const TestCase &tc) {
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
            std::cerr << tc.name << " seed=" << seed << " exceeded generation budget\n";
            return false;
        }

        worst = std::max(worst, result.best_fitness);

        if (verbose) {
            std::cout << std::fixed << std::setprecision(6)
                      << tc.name << " seed=" << seed
                      << " fitness=" << result.best_fitness
                      << " gen=" << result.budget_usage.generations
                      << " fevals=" << result.budget_usage.function_evaluations << "\n";
        }
    }

    if (worst > tc.max_fitness) {
        std::cerr << tc.name << " worst fitness " << worst << " exceeds limit " << tc.max_fitness << "\n";
        return false;
    }

    if (verbose) {
        std::cout << tc.name << " worst=" << worst << " ok\n";
    }

    return true;
}

} // namespace

int main() {
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;

    std::vector<TestCase> cases;

    // sphere 10d
    {
        TestCase tc;
        tc.name = "sphere";
        tc.problem = std::make_unique<wrappers::problems::SphereProblem>(10);
        tc.params.emplace("population_size", static_cast<std::int64_t>(120));
        tc.params.emplace("generations", static_cast<std::int64_t>(350));
        tc.params.emplace("scaling_factor", 0.7);
        tc.params.emplace("crossover_rate", 0.9);
        tc.budget.generations = 400;
        tc.budget.function_evaluations = 50000;
        tc.max_fitness = 5e-3;
        cases.push_back(std::move(tc));
    }

    // rosenbrock 6d
    // harder sowe allow more budget
    {
        TestCase tc;
        tc.name = "rosenbrock";
        tc.problem = std::make_unique<wrappers::problems::RosenbrockProblem>(6);
        tc.params.emplace("population_size", static_cast<std::int64_t>(150));
        tc.params.emplace("generations", static_cast<std::int64_t>(500));
        tc.params.emplace("scaling_factor", 0.6);
        tc.params.emplace("crossover_rate", 0.85);
        tc.budget.generations = 600;
        tc.budget.function_evaluations = 80000;
        tc.max_fitness = 1.0;
        cases.push_back(std::move(tc));
    }

    // run all test cases
    for (const auto &tc : cases) {
        if (!run(factory, tc)) {
            return 1;
        }
    }

    // test parameter validation
    // should reject invalid variant
    {
        auto algo = factory.create();
        core::ParameterSet invalid;
        invalid.emplace("variant", static_cast<std::int64_t>(0)); // out of range

        try {
            algo->configure(invalid);
            std::cerr << "expected ParameterValidationError for invalid variant\n";
            return 1;
        } catch (const core::ParameterValidationError &) {
            // expected
        }
    }

    return 0;
}

