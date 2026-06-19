#include "test_harness.hpp"
#include "test_utils.hpp"

#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
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
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem.evaluate({1.0, 1.0, 0.0}), 22.0, 1e-12),
                       "knapsack overweight selection objective is sum of values plus violation");
    }


    {
        std::vector<double> values{100.0, 100.0};
        std::vector<double> weights{1.0, 1.0};
        KnapsackProblem problem(values, weights, 1.95);
        const double single = problem.evaluate({1.0, 0.0});
        const double both = problem.evaluate({1.0, 1.0});
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(single, -100.0, 1e-12),
                       "knapsack feasible single-item objective is negative value");
        HPOEA_V2_CHECK(runner, both > single,
                       "knapsack infeasible both-items selection is strictly worse than feasible single-item");
        HPOEA_V2_CHECK(runner, both > 0.0,
                       "knapsack infeasible objective is positive (worse than every feasible objective <= 0)");
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

        struct BoundsCase {
            std::function<std::unique_ptr<hpoea::core::IProblem>()> make;
            const char *name;
            double lower;
            double upper;
        };
        const std::vector<BoundsCase> bounds_cases = {
            {[] { return std::make_unique<SphereProblem>(3); }, "Sphere", -5.0, 5.0},
            {[] { return std::make_unique<RosenbrockProblem>(3); }, "Rosenbrock", -5.0, 10.0},
            {[] { return std::make_unique<RastriginProblem>(3); }, "Rastrigin", -5.12, 5.12},
            {[] { return std::make_unique<AckleyProblem>(3); }, "Ackley", -32.768, 32.768},
            {[] { return std::make_unique<GriewankProblem>(3); }, "Griewank", -600.0, 600.0},
            {[] { return std::make_unique<SchwefelProblem>(3); }, "Schwefel", -500.0, 500.0},
            {[] { return std::make_unique<ZakharovProblem>(3); }, "Zakharov", -5.0, 10.0},
            {[] { return std::make_unique<StyblinskiTangProblem>(3); }, "StyblinskiTang", -5.0, 5.0},
        };
        for (const auto &c : bounds_cases) {
            const auto problem = c.make();
            check_bounds(*problem, c.name, c.lower, c.upper);
        }
    }


    {
        hpoea::config::ProblemParameterSet params;
        params.emplace("dimension", std::int64_t{3});
        params.emplace("lower_bond", -10.0);
        bool threw = false;
        std::string message;
        try {
            auto problem = hpoea::wrappers::problems::make_benchmark_problem("sphere", params);
            (void)problem;
        } catch (const std::exception &ex) {
            threw = true;
            message = ex.what();
        }
        HPOEA_V2_CHECK(runner, threw, "make_benchmark_problem rejects misspelled parameter key");
        HPOEA_V2_CHECK(runner, message.find("lower_bond") != std::string::npos,
                       "error message names the offending misspelled key");
    }


    {
        hpoea::config::ProblemParameterSet params;
        params.emplace("dimension", std::int64_t{4});
        params.emplace("lower_bound", -3.0);
        params.emplace("upper_bound", 3.0);
        auto problem = hpoea::wrappers::problems::make_benchmark_problem("sphere", params);
        HPOEA_V2_CHECK(runner, problem->dimension() == 4, "make_benchmark_problem builds sphere dim=4");
        HPOEA_V2_CHECK(runner, problem->metadata().id == "sphere",
                       "make_benchmark_problem sets correct metadata id");
        HPOEA_V2_CHECK(runner, hpoea::tests_v2::nearly_equal(problem->lower_bounds()[0], -3.0, 1e-12),
                       "make_benchmark_problem honors explicit lower_bound");
    }


    {
        hpoea::config::ProblemParameterSet params;
        params.emplace("values", std::vector<double>{10.0, 7.0});
        params.emplace("weights", std::vector<double>{5.0, 3.0});
        params.emplace("capcity", 6.0);
        bool threw = false;
        std::string message;
        try {
            auto problem = hpoea::wrappers::problems::make_benchmark_problem("knapsack", params);
            (void)problem;
        } catch (const std::exception &ex) {
            threw = true;
            message = ex.what();
        }
        HPOEA_V2_CHECK(runner, threw, "make_benchmark_problem rejects misspelled knapsack key");
        HPOEA_V2_CHECK(runner, message.find("capcity") != std::string::npos,
                       "knapsack error message names the offending misspelled key");
    }


    {
        // bounds omitted keeps each problem's canonical domain
        // never a uniform [-5, 5] fallback
        struct DefaultsCase { const char *type; double lower; double upper; };
        const std::vector<DefaultsCase> cases = {
            {"schwefel", -500.0, 500.0},
            {"ackley", -32.768, 32.768},
            {"griewank", -600.0, 600.0},
        };
        for (const auto &c : cases) {
            hpoea::config::ProblemParameterSet params;
            params.emplace("dimension", std::int64_t{2});
            auto problem = hpoea::wrappers::problems::make_benchmark_problem(c.type, params);
            const auto lb = problem->lower_bounds();
            const auto ub = problem->upper_bounds();
            bool ok = lb.size() == 2 && ub.size() == 2;
            for (std::size_t i = 0; i < lb.size() && i < ub.size(); ++i) {
                if (!hpoea::tests_v2::nearly_equal(lb[i], c.lower, 1e-12) ||
                    !hpoea::tests_v2::nearly_equal(ub[i], c.upper, 1e-12)) {
                    ok = false;
                }
            }
            HPOEA_V2_CHECK(runner, ok,
                           std::string("make_benchmark_problem ") + c.type +
                               " without bounds keeps its canonical default domain");
        }
    }


    {
        // exactly one of lower_bound/upper_bound must fail loudly
        // not pair with a default
        hpoea::config::ProblemParameterSet params;
        params.emplace("dimension", std::int64_t{2});
        params.emplace("lower_bound", -3.0);
        bool threw = false;
        std::string message;
        try {
            auto problem = hpoea::wrappers::problems::make_benchmark_problem("sphere", params);
            (void)problem;
        } catch (const std::invalid_argument &ex) {
            threw = true;
            message = ex.what();
        }
        HPOEA_V2_CHECK(runner, threw,
                       "make_benchmark_problem rejects lower_bound without upper_bound");
        HPOEA_V2_CHECK(runner, message.find("provided together") != std::string::npos,
                       "make_benchmark_problem error states bounds must be provided together");
    }


    return runner.summarize("benchmark_problem_tests");
}
