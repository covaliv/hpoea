#include "test_harness.hpp"
#include "test_utils.hpp"

#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <limits>

int main() {
    hpoea::tests_v2::TestRunner runner;
    using namespace hpoea::wrappers::problems;


    {
        SphereProblem problem(4);
        std::vector<double> optimum(4, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, 0.0, 1e-12),
                       "sphere optimum evaluates to 0");
        HPOEA_V2_CHECK(runner, problem.metadata().id == "sphere",
                       "sphere metadata id set");
    }


    {
        RosenbrockProblem problem(3);
        std::vector<double> optimum(3, 1.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, std::fabs(value) < 1e-9,
                       "rosenbrock optimum evaluates near 0");
    }


    {
        RastriginProblem problem(5);
        std::vector<double> optimum(5, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, 0.0, 1e-12),
                       "rastrigin optimum evaluates to 0");
    }


    {
        AckleyProblem problem(2);
        std::vector<double> optimum(2, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, std::fabs(value) < 1e-9,
                       "ackley optimum evaluates near 0");
    }


    {
        GriewankProblem problem(3);
        std::vector<double> optimum(3, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, 0.0, 1e-12),
                       "griewank optimum evaluates to 0");
    }


    {
        SchwefelProblem problem(2);
        std::vector<double> optimum(2, 420.968746);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, std::fabs(value) < 1e-4,
                       "schwefel optimum evaluates near 0");
    }


    {
        ZakharovProblem problem(3);
        std::vector<double> optimum(3, 0.0);
        const double value = problem.evaluate(optimum);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, 0.0, 1e-12),
                       "zakharov optimum evaluates to 0");
    }


    {
        StyblinskiTangProblem problem(2);
        std::vector<double> optimum(2, -2.903534);
        const double value = problem.evaluate(optimum);
        const double expected = -39.16599 * 2.0;
        HPOEA_V2_CHECK(runner, std::fabs(value - expected) < 1e-3,
                       "styblinski-tang optimum matches expected value");
    }


    {
        std::vector<double> values{10.0, 7.0, 3.0};
        std::vector<double> weights{5.0, 3.0, 1.0};
        KnapsackProblem problem(values, weights, 6.0);
        std::vector<double> selection{0.0, 1.0, 1.0};
        const double value = problem.evaluate(selection);
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(value, -10.0, 1e-12),
                       "knapsack objective is negative total value when within capacity");
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
        auto check_bounds = [&](const hpoea::core::IProblem &prob, const std::string &name) {
            auto lb = prob.lower_bounds();
            auto ub = prob.upper_bounds();
            HPOEA_V2_CHECK(runner, lb.size() == prob.dimension(),
                           name + " lower_bounds size matches dimension");
            HPOEA_V2_CHECK(runner, ub.size() == prob.dimension(),
                           name + " upper_bounds size matches dimension");
            bool all_valid = true;
            for (std::size_t i = 0; i < lb.size(); ++i) {
                if (lb[i] >= ub[i]) { all_valid = false; break; }
            }
            HPOEA_V2_CHECK(runner, all_valid,
                           name + " all lower bounds < upper bounds");
        };

        SphereProblem sphere(3);
        check_bounds(sphere, "Sphere");

        RosenbrockProblem rosenbrock(3);
        check_bounds(rosenbrock, "Rosenbrock");

        RastriginProblem rastrigin(3);
        check_bounds(rastrigin, "Rastrigin");

        AckleyProblem ackley(3);
        check_bounds(ackley, "Ackley");

        GriewankProblem griewank(3);
        check_bounds(griewank, "Griewank");

        SchwefelProblem schwefel(3);
        check_bounds(schwefel, "Schwefel");

        ZakharovProblem zakharov(3);
        check_bounds(zakharov, "Zakharov");

        StyblinskiTangProblem styblinskitang(3);
        check_bounds(styblinskitang, "StyblinskiTang");
    }


    {

        SphereProblem sphere(3);
        HPOEA_V2_CHECK(runner, std::abs(sphere.evaluate({0.0, 0.0, 0.0})) < 1e-10,
                       "Sphere evaluate at optimum = 0.0");


        RosenbrockProblem rosenbrock(3);
        HPOEA_V2_CHECK(runner, std::abs(rosenbrock.evaluate({1.0, 1.0, 1.0})) < 1e-10,
                       "Rosenbrock evaluate at optimum = 0.0");


        RastriginProblem rastrigin(3);
        HPOEA_V2_CHECK(runner, std::abs(rastrigin.evaluate({0.0, 0.0, 0.0})) < 1e-10,
                       "Rastrigin evaluate at optimum = 0.0");
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
