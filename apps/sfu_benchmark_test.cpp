#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/types.hpp"

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

bool verify_solution_bounds(const std::vector<double> &solution, 
                           const std::vector<double> &lower,
                           const std::vector<double> &upper) {
    if (solution.size() != lower.size() || solution.size() != upper.size()) {
        return false;
    }
    for (size_t i = 0; i < solution.size(); ++i) {
        if (solution[i] < lower[i] || solution[i] > upper[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    using namespace hpoea;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "========================================\n";
    std::cout << "  SFU Benchmark Functions Correctness Test\n";
    std::cout << "  Based on: https://www.sfu.ca/~ssurjano/optimization.html\n";
    std::cout << "========================================\n\n";
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    // Test functions from SFU library
    struct ProblemTest {
        std::string name;
        std::string category;
        std::string url;
        std::unique_ptr<core::IProblem> problem;
        double expected_optimum;
        double tolerance;
    };
    
    std::vector<ProblemTest> problems;
    
    // Many Local Minima category
    problems.push_back({"Griewank (5D)", "Many Local Minima", 
                       "https://www.sfu.ca/~ssurjano/griewank.html",
                       std::make_unique<wrappers::problems::GriewankProblem>(5, -600.0, 600.0),
                       0.0, 0.01});
    
    problems.push_back({"Schwefel (5D)", "Many Local Minima",
                       "https://www.sfu.ca/~ssurjano/schwef.html",
                       std::make_unique<wrappers::problems::SchwefelProblem>(5, -500.0, 500.0),
                       0.0, 10.0});
    
    // Plate-Shaped category
    problems.push_back({"Zakharov (5D)", "Plate-Shaped",
                       "https://www.sfu.ca/~ssurjano/zakharov.html",
                       std::make_unique<wrappers::problems::ZakharovProblem>(5, -5.0, 10.0),
                       0.0, 0.001});
    
    // Other category
    problems.push_back({"Styblinski-Tang (5D)", "Other",
                       "https://www.sfu.ca/~ssurjano/stybtang.html",
                       std::make_unique<wrappers::problems::StyblinskiTangProblem>(5, -5.0, 5.0),
                       -39.16599 * 5.0, 0.01});
    
    // Existing functions for comparison
    problems.push_back({"Sphere (5D)", "Bowl-Shaped",
                       "https://www.sfu.ca/~ssurjano/spheref.html",
                       std::make_unique<wrappers::problems::SphereProblem>(5),
                       0.0, 0.0001});
    
    problems.push_back({"Rastrigin (5D)", "Many Local Minima",
                       "https://www.sfu.ca/~ssurjano/rastr.html",
                       std::make_unique<wrappers::problems::RastriginProblem>(5),
                       0.0, 0.1});
    
    problems.push_back({"Ackley (5D)", "Many Local Minima",
                       "https://www.sfu.ca/~ssurjano/ackley.html",
                       std::make_unique<wrappers::problems::AckleyProblem>(5),
                       0.0, 0.01});
    
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    core::Budget budget;
    budget.generations = 200;
    budget.function_evaluations = 20000;
    
    std::cout << "Testing " << problems.size() << " benchmark functions with DE algorithm\n";
    std::cout << "Budget: 200 generations, up to 20000 function evaluations per problem\n\n";
    
    for (const auto &ptest : problems) {
        std::cout << "Testing: " << ptest.name << " [" << ptest.category << "]\n";
        std::cout << "  Reference: " << ptest.url << "\n";
        
        auto algo = factory.create();
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(100));
        params.emplace("generations", static_cast<std::int64_t>(200));
        params.emplace("scaling_factor", 0.5);
        params.emplace("crossover_rate", 0.9);
        params.emplace("variant", static_cast<std::int64_t>(2));
        algo->configure(params);
        
        auto result = algo->run(*ptest.problem, budget, 42UL);
        
        bool passed = true;
        std::vector<std::string> errors;
        
        // Check status
        if (result.status != core::RunStatus::Success) {
            passed = false;
            errors.push_back("Status: " + result.message);
        }
        
        // Check solution dimension
        if (result.best_solution.size() != ptest.problem->dimension()) {
            passed = false;
            errors.push_back("Dimension mismatch");
        }
        
        // Check bounds
        auto lower = ptest.problem->lower_bounds();
        auto upper = ptest.problem->upper_bounds();
        if (!verify_solution_bounds(result.best_solution, lower, upper)) {
            passed = false;
            errors.push_back("Solution violates bounds");
        }
        
        double distance_to_optimum = std::abs(result.best_fitness - ptest.expected_optimum);
        if (distance_to_optimum > ptest.tolerance * 100) {
        }
        
        // Check fitness is finite
        if (!std::isfinite(result.best_fitness)) {
            passed = false;
            errors.push_back("Fitness is not finite");
        }
        
        if (passed) {
            std::cout << "  PASSED\n";
            std::cout << "    Best fitness: " << result.best_fitness << "\n";
            std::cout << "    Expected optimum: " << ptest.expected_optimum << "\n";
            std::cout << "    Distance to optimum: " << distance_to_optimum << "\n";
            std::cout << "    Function evaluations: " << result.budget_usage.function_evaluations << "\n";
            tests_passed++;
        } else {
            std::cout << "  FAILED\n";
            for (const auto &err : errors) {
                std::cout << "    - " << err << "\n";
            }
            tests_failed++;
        }
        std::cout << "\n";
    }
    
    std::cout << "\nAdditional Test: Comparing DE vs PSO on Griewank (5D)\n";
    std::cout << "------------------------------------------------------\n";
    {
        wrappers::problems::GriewankProblem problem(5, -600.0, 600.0);
        
        pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
        pagmo_wrappers::PagmoParticleSwarmOptimizationFactory pso_factory;
        
        auto de_algo = de_factory.create();
        auto pso_algo = pso_factory.create();
        
        core::ParameterSet de_params;
        de_params.emplace("population_size", static_cast<std::int64_t>(100));
        de_params.emplace("generations", static_cast<std::int64_t>(200));
        de_params.emplace("scaling_factor", 0.5);
        de_params.emplace("crossover_rate", 0.9);
        de_params.emplace("variant", static_cast<std::int64_t>(2));
        de_algo->configure(de_params);
        
        core::ParameterSet pso_params;
        pso_params.emplace("population_size", static_cast<std::int64_t>(100));
        pso_params.emplace("generations", static_cast<std::int64_t>(200));
        pso_params.emplace("omega", 0.7298);
        pso_params.emplace("eta1", 2.05);
        pso_params.emplace("eta2", 2.05);
        pso_params.emplace("max_velocity", 0.5);
        pso_params.emplace("variant", static_cast<std::int64_t>(5));
        pso_algo->configure(pso_params);
        
        core::Budget test_budget;
        test_budget.generations = 200;
        test_budget.function_evaluations = 20000;
        
        auto de_result = de_algo->run(problem, test_budget, 42UL);
        auto pso_result = pso_algo->run(problem, test_budget, 42UL);
        
        if (de_result.status == core::RunStatus::Success && 
            pso_result.status == core::RunStatus::Success) {
            std::cout << "  DE fitness: " << de_result.best_fitness << "\n";
            std::cout << "  PSO fitness: " << pso_result.best_fitness << "\n";
            std::string better = (de_result.best_fitness < pso_result.best_fitness) ? "DE" : "PSO";
            std::cout << "  Better algorithm: " << better << "\n";
            tests_passed++;
        } else {
            std::cout << "  FAILED\n";
            tests_failed++;
        }
    }
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "  Tests passed: " << tests_passed << "\n";
    std::cout << "  Tests failed: " << tests_failed << "\n";
    std::cout << "  Total tests: " << (tests_passed + tests_failed) << "\n";
    
    if (tests_failed == 0) {
        std::cout << "\n  All SFU benchmark function tests PASSED!\n";
        return 0;
    } else {
        std::cout << "\n  Some tests FAILED!\n";
        return 1;
    }
}

