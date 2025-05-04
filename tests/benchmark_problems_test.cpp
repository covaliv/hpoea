#include "hpoea/core/problem.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main() {
    using namespace hpoea;

    const bool verbose = [] {
        if (const char *flag = std::getenv("HPOEA_LOG_RESULTS")) {
            return std::string_view{flag} == "1";
        }
        return false;
    }();

    struct TestCase {
        std::string name;
        std::unique_ptr<core::IProblem> problem;
        std::vector<double> expected_optimum;
        double tolerance;
    };

    std::vector<TestCase> test_cases;

    // Sphere: optimum at (0, 0, ...)
    {
        auto problem = std::make_unique<wrappers::problems::SphereProblem>(5);
        std::vector<double> optimum(5, 0.0);
        test_cases.push_back({"sphere", std::move(problem), optimum, 1e-10});
    }

    // Rosenbrock: optimum near (1, 1, ...)
    {
        auto problem = std::make_unique<wrappers::problems::RosenbrockProblem>(5);
        std::vector<double> optimum(5, 1.0);
        test_cases.push_back({"rosenbrock", std::move(problem), optimum, 1e-3});
    }

    // Rastrigin: optimum at (0, 0, ...)
    {
        auto problem = std::make_unique<wrappers::problems::RastriginProblem>(5);
        std::vector<double> optimum(5, 0.0);
        test_cases.push_back({"rastrigin", std::move(problem), optimum, 1e-10});
    }

    // Ackley: optimum at (0, 0, ...)
    {
        auto problem = std::make_unique<wrappers::problems::AckleyProblem>(5);
        std::vector<double> optimum(5, 0.0);
        test_cases.push_back({"ackley", std::move(problem), optimum, 1e-10});
    }

    bool all_passed = true;

    for (const auto &test_case : test_cases) {
        const auto &problem = test_case.problem;

        // Test metadata
        if (problem->metadata().id.empty()) {
            std::cerr << "Test case '" << test_case.name << "' has empty metadata ID" << '\n';
            all_passed = false;
            continue;
        }

        if (problem->dimension() != test_case.expected_optimum.size()) {
            std::cerr << "Test case '" << test_case.name << "' dimension mismatch" << '\n';
            all_passed = false;
            continue;
        }

        // Test bounds
        const auto lower = problem->lower_bounds();
        const auto upper = problem->upper_bounds();
        if (lower.size() != problem->dimension() || upper.size() != problem->dimension()) {
            std::cerr << "Test case '" << test_case.name << "' bounds size mismatch" << '\n';
            all_passed = false;
            continue;
        }

        // Test evaluation at optimum
        const double optimum_value = problem->evaluate(test_case.expected_optimum);
        if (std::isnan(optimum_value) || std::isinf(optimum_value)) {
            std::cerr << "Test case '" << test_case.name << "' evaluation at optimum is invalid" << '\n';
            all_passed = false;
            continue;
        }

        // Test evaluation at random point
        std::vector<double> random_point(problem->dimension());
        for (std::size_t i = 0; i < problem->dimension(); ++i) {
            random_point[i] = (lower[i] + upper[i]) / 2.0;
        }
        const double random_value = problem->evaluate(random_point);
        if (std::isnan(random_value) || std::isinf(random_value)) {
            std::cerr << "Test case '" << test_case.name << "' evaluation at random point is invalid" << '\n';
            all_passed = false;
            continue;
        }

        if (optimum_value > random_value + test_case.tolerance) {
            std::cerr << "Test case '" << test_case.name << "' optimum value (" << optimum_value
                      << ") not better than random point (" << random_value << ")" << '\n';
            all_passed = false;
            continue;
        }

        if (verbose) {
            std::cout << std::fixed << std::setprecision(10)
                      << "test=" << test_case.name
                      << ", dimension=" << problem->dimension()
                      << ", optimum_value=" << optimum_value
                      << ", random_value=" << random_value << '\n';
        }
    }

    // Test error handling
    {
        wrappers::problems::SphereProblem problem(3);
        std::vector<double> wrong_size(5, 0.0);
        try {
            (void)problem.evaluate(wrong_size);
            std::cerr << "Expected exception for wrong dimension" << '\n';
            all_passed = false;
        } catch (const std::exception &) {
            // Expected - can be runtime_error or any exception
        }
    }

    return all_passed ? 0 : 1;
}

