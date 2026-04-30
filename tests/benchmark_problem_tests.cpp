#include "test_harness.hpp"
#include "test_utils.hpp"

#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

int main() {
    hpoea::tests_v2::TestRunner runner;
    using namespace hpoea::wrappers::problems;


    {
        SphereProblem problem(4);
        std::vector<double> optimum(4, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, 0.0, 1e-12),
                       "sphere optimum evaluates to 0");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({1.0, -2.0, 3.0, -4.0}), 30.0, 1e-12),
                       "sphere formula sums squared coordinates");
        HPOEA_V2_CHECK(runner, problem.metadata().id == "sphere",
                       "sphere metadata id set");
    }


    {
        RosenbrockProblem problem(3);
        std::vector<double> optimum(3, 1.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, std::fabs(value) < 1e-9,
                       "rosenbrock optimum evaluates near 0");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({1.0, 2.0, 3.0}), 201.0, 1e-12),
                       "rosenbrock formula sums adjacent valley terms");
    }


    {
        RastriginProblem problem(5);
        std::vector<double> optimum(5, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, 0.0, 1e-12),
                       "rastrigin optimum evaluates to 0");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({1.0, 0.0, -1.0, 0.0, 2.0}), 6.0, 1e-12),
                       "rastrigin integer coordinates reduce to squared sum");
    }


    {
        AckleyProblem problem(2);
        std::vector<double> optimum(2, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, std::fabs(value) < 1e-9,
                       "ackley optimum evaluates near 0");
        const double expected = -20.0 * std::exp(-0.2) - std::exp(1.0) + 20.0 + std::exp(1.0);
        HPOEA_V2_CHECK(runner, std::fabs(problem.evaluate({1.0, -1.0}) - expected) < 1e-12,
                       "ackley formula handles nonzero symmetric point");
    }


    {
        GriewankProblem problem(3);
        std::vector<double> optimum(3, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, 0.0, 1e-12),
                       "griewank optimum evaluates to 0");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({0.0, 0.0, 2.0}), 0.001 - std::cos(2.0 / std::sqrt(3.0)) + 1.0, 1e-12),
                       "griewank formula combines sum and product terms");
    }


    {
        SchwefelProblem problem(2);
        std::vector<double> optimum(2, 420.968746);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, std::fabs(value) < 1e-4,
                       "schwefel optimum evaluates near 0");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({0.0, 0.0}), 837.9657745448678, 1e-12),
                       "schwefel zero point equals dimension-scaled alpha");
    }


    {
        ZakharovProblem problem(3);
        std::vector<double> optimum(3, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, 0.0, 1e-12),
                       "zakharov optimum evaluates to 0");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({1.0, 0.0, 0.0}), 1.3125, 1e-12),
                       "zakharov formula includes quadratic and weighted sum terms");
    }


    {
        StyblinskiTangProblem problem(2);
        std::vector<double> optimum(2, -2.903534);
        const double value = problem.evaluate(optimum);
        const double expected = -39.16599 * 2.0;
        HPOEA_V2_CHECK(runner, std::fabs(value - expected) < 1e-3,
                       "styblinski-tang optimum matches expected value");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({0.0, 1.0}), -5.0, 1e-12),
                       "styblinski-tang formula evaluates simple mixed point");
    }


    {
        std::vector<double> values{10.0, 7.0, 3.0};
        std::vector<double> weights{5.0, 3.0, 1.0};
        KnapsackProblem problem(values, weights, 6.0);
        std::vector<double> selection{0.0, 1.0, 1.0};
        const double value = problem.evaluate(selection);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, -10.0, 1e-12),
                       "knapsack objective is negative total value when within capacity");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({1.0, 1.0, 0.0}), 23.0, 1e-12),
                       "knapsack overweight selection applies capacity penalty");
    }


    {
        SphereProblem problem(3);
        bool threw = false;
        try {
            std::vector<double> wrong(4, 0.0);
            (void)problem.evaluate(wrong);
        } catch (const std::exception &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "dimension mismatch throws");
    }


    {
        auto check_bounds = [&](const hpoea::core::IProblem &prob, const std::string &name,
                                double expected_lower, double expected_upper) {
            auto lb = prob.lower_bounds();
            auto ub = prob.upper_bounds();
            HPOEA_V2_CHECK(runner, lb.size() == prob.dimension(),
                           name + " lower_bounds size matches dimension");
            HPOEA_V2_CHECK(runner, ub.size() == prob.dimension(),
                           name + " upper_bounds size matches dimension");
            bool all_match = lb.size() == ub.size();
            for (std::size_t i = 0; i < lb.size() && i < ub.size(); ++i) {
                if (!hpoea::tests_v2::nearly_equal(lb[i], expected_lower, 1e-12) ||
                    !hpoea::tests_v2::nearly_equal(ub[i], expected_upper, 1e-12) ||
                    lb[i] >= ub[i]) {
                    all_match = false;
                    break;
                }
            }
            HPOEA_V2_CHECK(runner, all_match,
                           name + " bounds match expected defaults");
        };

        SphereProblem sphere(3);
        check_bounds(sphere, "Sphere", -5.0, 5.0);

        RosenbrockProblem rosenbrock(3);
        check_bounds(rosenbrock, "Rosenbrock", -5.0, 10.0);

        RastriginProblem rastrigin(3);
        check_bounds(rastrigin, "Rastrigin", -5.12, 5.12);

        AckleyProblem ackley(3);
        check_bounds(ackley, "Ackley", -32.768, 32.768);

        GriewankProblem griewank(3);
        check_bounds(griewank, "Griewank", -600.0, 600.0);

        SchwefelProblem schwefel(3);
        check_bounds(schwefel, "Schwefel", -500.0, 500.0);

        ZakharovProblem zakharov(3);
        check_bounds(zakharov, "Zakharov", -5.0, 10.0);

        StyblinskiTangProblem styblinskitang(3);
        check_bounds(styblinskitang, "StyblinskiTang", -5.0, 5.0);
    }


    {
        SphereProblem s1(1);
        SphereProblem s5(5);
        SphereProblem s10(10);
        HPOEA_V2_CHECK(runner, s1.dimension() == 1, "Sphere dim=1");
        HPOEA_V2_CHECK(runner, s5.dimension() == 5, "Sphere dim=5");
        HPOEA_V2_CHECK(runner, s10.dimension() == 10, "Sphere dim=10");
        HPOEA_V2_CHECK(runner, s1.lower_bounds().size() == 1, "Sphere dim=1 bounds");
        HPOEA_V2_CHECK(runner, s10.lower_bounds().size() == 10, "Sphere dim=10 bounds");
    }

    return runner.summarize("benchmark_problem_tests");
}
