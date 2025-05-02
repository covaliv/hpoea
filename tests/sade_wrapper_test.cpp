#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <variant>
#include <vector>

int main() {
    using namespace hpoea;

    const bool verbose = [] {
        if (const char *flag = std::getenv("HPOEA_LOG_RESULTS")) {
            return std::string_view{flag} == "1";
        }
        return false;
    }();

    pagmo_wrappers::PagmoSelfAdaptiveDEFactory factory;
    const std::vector<unsigned long> seeds{42UL, 1337UL, 2024UL, 9001UL};

    struct TestCase {
        std::string name;
        std::function<std::unique_ptr<hpoea::core::IProblem>()> make_problem;
        core::ParameterSet parameters;
        core::Budget budget;
        double max_allowed_fitness{0.0};
    };

    const auto run_test_case = [&](const TestCase &test_case) -> bool {
        double worst_fitness = 0.0;

        for (const auto seed : seeds) {
            auto problem = test_case.make_problem();
            if (!problem) {
                std::cerr << "Unable to construct problem for test case '" << test_case.name << "'" << '\n';
                return false;
            }

            auto algorithm = factory.create();
            algorithm->configure(test_case.parameters);

            const auto result = algorithm->run(*problem, test_case.budget, seed);

            if (result.status != core::RunStatus::Success) {
                std::cerr << "Test case '" << test_case.name << "' failed for seed " << seed
                          << " with error: " << result.message << '\n';
                return false;
            }

            if (result.best_solution.size() != problem->dimension()) {
                std::cerr << "Test case '" << test_case.name << "' returned invalid solution size for seed " << seed
                          << '\n';
                return false;
            }

            if (result.budget_usage.generations == 0
                || (test_case.budget.generations.has_value()
                    && result.budget_usage.generations > test_case.budget.generations.value())) {
                std::cerr << "Test case '" << test_case.name << "' used unexpected generation count for seed " << seed
                          << '\n';
                return false;
            }

            worst_fitness = std::max(worst_fitness, result.best_fitness);

            if (verbose) {
                std::cout << std::fixed << std::setprecision(6)
                          << "test=" << test_case.name << ", seed=" << seed
                          << ", best_fitness=" << result.best_fitness
                          << ", generations=" << result.budget_usage.generations
                          << ", fevals=" << result.budget_usage.function_evaluations << '\n';
            }
        }

        if (worst_fitness > test_case.max_allowed_fitness) {
            std::cerr << "Test case '" << test_case.name << "' worst fitness too large: " << worst_fitness << '\n';
            return false;
        }

        if (verbose) {
            std::cout << "test=" << test_case.name << ", worst_fitness=" << worst_fitness << '\n';
        }

        return true;
    };

    std::vector<TestCase> test_cases;

    {
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(50));
        params.emplace("generations", static_cast<std::int64_t>(200));
        params.emplace("variant", static_cast<std::int64_t>(2));
        params.emplace("variant_adptv", static_cast<std::int64_t>(1));

        core::Budget budget;
        budget.generations = 250;
        budget.function_evaluations = 15000;

        test_cases.push_back(TestCase{
            "sphere",
            [] { return std::make_unique<wrappers::problems::SphereProblem>(10); },
            std::move(params),
            budget,
            1e-2});
    }

    {
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(60));
        params.emplace("generations", static_cast<std::int64_t>(300));
        params.emplace("variant", static_cast<std::int64_t>(3));
        params.emplace("variant_adptv", static_cast<std::int64_t>(2));

        core::Budget budget;
        budget.generations = 350;
        budget.function_evaluations = 25000;

        test_cases.push_back(TestCase{
            "rosenbrock",
            [] { return std::make_unique<wrappers::problems::RosenbrockProblem>(6); },
            std::move(params),
            budget,
            5.0});
    }

    for (const auto &test_case : test_cases) {
        if (!run_test_case(test_case)) {
            return 1;
        }
    }

    // Test parameter validation
    auto algorithm = factory.create();
    try {
        core::ParameterSet invalid;
        invalid.emplace("variant", static_cast<std::int64_t>(0)); // Out of range
        algorithm->configure(invalid);
        std::cerr << "Expected ParameterValidationError for invalid variant" << '\n';
        return 5;
    } catch (const core::ParameterValidationError &) {
        // Expected
    }

    return 0;
}

