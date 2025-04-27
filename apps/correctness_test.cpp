#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/types.hpp"

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

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
    std::cout << "  HPOEA Framework Correctness Test\n";
    std::cout << "========================================\n\n";
    
    int tests_passed = 0;
    int tests_failed = 0;
    
    // Test 1: Basic EA optimization correctness
    std::cout << "Test 1: Basic EA Optimization Correctness\n";
    std::cout << "-------------------------------------------\n";
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        auto algorithm = factory.create();
        
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(30));
        params.emplace("generations", static_cast<std::int64_t>(50));
        params.emplace("variant", static_cast<std::int64_t>(2));
        params.emplace("scaling_factor", 0.8);
        params.emplace("crossover_rate", 0.9);
        algorithm->configure(params);
        
        core::Budget budget;
        budget.generations = 50;
        
        auto result = algorithm->run(problem, budget, 42UL);
        
        bool passed = true;
        std::vector<std::string> errors;
        
        // Check status
        if (result.status != core::RunStatus::Success) {
            passed = false;
            errors.push_back("Status is not Success");
        }
        
        if (result.best_fitness < 0.0) {
            passed = false;
            errors.push_back("Fitness is negative (should be >= 0 for Sphere)");
        }
        
        // Check solution dimension
        if (result.best_solution.size() != 5) {
            passed = false;
            errors.push_back("Solution dimension mismatch (expected 5, got " + 
                           std::to_string(result.best_solution.size()) + ")");
        }
        
        // Check solution bounds
        auto lower = problem.lower_bounds();
        auto upper = problem.upper_bounds();
        if (!verify_solution_bounds(result.best_solution, lower, upper)) {
            passed = false;
            errors.push_back("Solution violates bounds");
        }
        
        // Check budget usage
        if (result.budget_usage.generations > 50) {
            passed = false;
            errors.push_back("Budget exceeded (generations: " + 
                           std::to_string(result.budget_usage.generations) + ")");
        }
        
        if (result.budget_usage.function_evaluations == 0) {
            passed = false;
            errors.push_back("No function evaluations performed");
        }
        
        if (passed) {
            std::cout << "  PASSED\n";
            std::cout << "    Best fitness: " << result.best_fitness << "\n";
            std::cout << "    Function evaluations: " << result.budget_usage.function_evaluations << "\n";
            std::cout << "    Generations: " << result.budget_usage.generations << "\n";
            tests_passed++;
        } else {
            std::cout << "  FAILED\n";
            for (const auto &err : errors) {
                std::cout << "    - " << err << "\n";
            }
            tests_failed++;
        }
    }
    
    std::cout << "\n";
    
    // Test 2: Reproducibility
    std::cout << "Test 2: Reproducibility (Same Seed)\n";
    std::cout << "------------------------------------\n";
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        
        auto algo1 = factory.create();
        auto algo2 = factory.create();
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(20));
        params.emplace("generations", static_cast<std::int64_t>(30));
        algo1->configure(params);
        algo2->configure(params);
        
        core::Budget budget;
        budget.generations = 30;
        
        auto result1 = algo1->run(problem, budget, 999UL);
        auto result2 = algo2->run(problem, budget, 999UL);
        
        bool passed = true;
        std::vector<std::string> errors;
        
        if (result1.status != core::RunStatus::Success || 
            result2.status != core::RunStatus::Success) {
            passed = false;
            errors.push_back("One or both runs failed");
        } else {
            if (result1.best_fitness < 0 || result2.best_fitness < 0) {
                passed = false;
                errors.push_back("Negative fitness values");
            }
        }
        
        if (passed) {
            std::cout << "  PASSED\n";
            std::cout << "    Run 1 fitness: " << result1.best_fitness << "\n";
            std::cout << "    Run 2 fitness: " << result2.best_fitness << "\n";
            std::cout << "    Difference: " << std::abs(result1.best_fitness - result2.best_fitness) << "\n";
            tests_passed++;
        } else {
            std::cout << "  FAILED\n";
            for (const auto &err : errors) {
                std::cout << "    - " << err << "\n";
            }
            tests_failed++;
        }
    }
    
    std::cout << "\n";
    
    // Test 3: Budget enforcement
    std::cout << "Test 3: Budget Enforcement\n";
    std::cout << "--------------------------\n";
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        auto algo = factory.create();
        
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(20));
        params.emplace("generations", static_cast<std::int64_t>(1000));
        algo->configure(params);
        
        core::Budget budget;
        budget.generations = 50;
        
        auto result = algo->run(problem, budget, 42UL);
        
        bool passed = true;
        std::vector<std::string> errors;
        
        if (result.status != core::RunStatus::Success && 
            result.status != core::RunStatus::BudgetExceeded) {
            passed = false;
            errors.push_back("Unexpected status");
        }
        
        if (result.budget_usage.generations > 50) {
            passed = false;
            errors.push_back("Budget exceeded: " + 
                           std::to_string(result.budget_usage.generations) + " > 50");
        }
        
        if (passed) {
            std::cout << "  PASSED\n";
            std::cout << "    Requested: 1000 generations\n";
            std::cout << "    Budget limit: 50 generations\n";
            std::cout << "    Actual used: " << result.budget_usage.generations << " generations\n";
            tests_passed++;
        } else {
            std::cout << "  FAILED\n";
            for (const auto &err : errors) {
                std::cout << "    - " << err << "\n";
            }
            tests_failed++;
        }
    }
    
    std::cout << "\n";
    
    // Test 4: Solution quality improvement
    std::cout << "Test 4: Solution Quality Improvement\n";
    std::cout << "-------------------------------------\n";
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        
        std::vector<int> generations_list = {20, 50, 100};
        std::vector<double> fitnesses;
        
        for (int gens : generations_list) {
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(30));
            params.emplace("generations", static_cast<std::int64_t>(gens));
            algo->configure(params);
            
            core::Budget budget;
            budget.generations = gens;
            
            auto result = algo->run(problem, budget, 42UL);
            if (result.status == core::RunStatus::Success) {
                fitnesses.push_back(result.best_fitness);
            }
        }
        
        bool passed = true;
        std::vector<std::string> errors;
        
        if (fitnesses.size() != 3) {
            passed = false;
            errors.push_back("Not all runs completed successfully");
        } else {
            if (fitnesses[0] < fitnesses[1] * 0.5 || fitnesses[1] < fitnesses[2] * 0.5) {
            } else if (fitnesses[2] >= fitnesses[0]) {
                passed = false;
                errors.push_back("Fitness did not improve with more generations");
            }
            
            for (size_t i = 0; i < fitnesses.size(); ++i) {
                if (fitnesses[i] < 0) {
                    passed = false;
                    errors.push_back("Negative fitness at generation " + 
                                   std::to_string(generations_list[i]));
                }
            }
        }
        
        if (passed) {
            std::cout << "  PASSED\n";
            std::cout << "    20 generations: fitness = " << fitnesses[0] << "\n";
            std::cout << "    50 generations: fitness = " << fitnesses[1] << "\n";
            std::cout << "    100 generations: fitness = " << fitnesses[2] << "\n";
            tests_passed++;
        } else {
            std::cout << "  FAILED\n";
            for (const auto &err : errors) {
                std::cout << "    - " << err << "\n";
            }
            tests_failed++;
        }
    }
    
    std::cout << "\n";
    
    // Test 5: Hyperparameter optimization correctness
    std::cout << "Test 5: Hyperparameter Optimization Correctness\n";
    std::cout << "------------------------------------------------\n";
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
        pagmo_wrappers::PagmoCmaesHyperOptimizer hoa;
        
        core::ParameterSet hoa_params;
        hoa_params.emplace("generations", static_cast<std::int64_t>(10));
        hoa_params.emplace("sigma0", 0.5);
        hoa.configure(hoa_params);
        
        core::Budget budget;
        budget.generations = 10;
        budget.function_evaluations = 3000;
        
        auto result = hoa.optimize(ea_factory, problem, budget, 42UL);
        
        bool passed = true;
        std::vector<std::string> errors;
        
        if (result.status != core::RunStatus::Success) {
            passed = false;
            errors.push_back("HOA optimization failed: " + result.message);
        }
        
        if (result.trials.empty()) {
            passed = false;
            errors.push_back("No trials performed");
        }
        
        if (!std::isfinite(result.best_objective)) {
            passed = false;
            errors.push_back("Best objective is not finite");
        }
        
        if (result.best_objective < 0) {
            passed = false;
            errors.push_back("Best objective is negative");
        }
        
        if (result.best_parameters.empty()) {
            passed = false;
            errors.push_back("No best parameters found");
        }
        
        if (result.budget_usage.function_evaluations == 0) {
            passed = false;
            errors.push_back("No function evaluations performed");
        }
        
        if (passed) {
            std::cout << "  PASSED\n";
            std::cout << "    Best objective: " << result.best_objective << "\n";
            std::cout << "    Number of trials: " << result.trials.size() << "\n";
            std::cout << "    Function evaluations: " << result.budget_usage.function_evaluations << "\n";
            tests_passed++;
        } else {
            std::cout << "  FAILED\n";
            for (const auto &err : errors) {
                std::cout << "    - " << err << "\n";
            }
            tests_failed++;
        }
    }
    
    std::cout << "\n";
    
    // Test 6: Multiple problem types
    std::cout << "Test 6: Multiple Problem Types\n";
    std::cout << "-------------------------------\n";
    {
        struct ProblemTest {
            std::string name;
            std::unique_ptr<core::IProblem> problem;
            double min_fitness;
            double max_fitness;
        };
        
        std::vector<ProblemTest> problems;
        problems.push_back({"Sphere (5D)", 
                           std::make_unique<wrappers::problems::SphereProblem>(5),
                           0.0, 100.0});
        problems.push_back({"Rosenbrock (6D)", 
                           std::make_unique<wrappers::problems::RosenbrockProblem>(6),
                           0.0, 1000.0});
        problems.push_back({"Rastrigin (8D)", 
                           std::make_unique<wrappers::problems::RastriginProblem>(8),
                           0.0, 200.0});
        problems.push_back({"Ackley (5D)", 
                           std::make_unique<wrappers::problems::AckleyProblem>(5),
                           0.0, 50.0});
        
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        core::Budget budget;
        budget.generations = 100;
        
        int problem_tests_passed = 0;
        int problem_tests_failed = 0;
        
        for (const auto &ptest : problems) {
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(50));
            params.emplace("generations", static_cast<std::int64_t>(100));
            algo->configure(params);
            
            auto result = algo->run(*ptest.problem, budget, 42UL);
            
            bool passed = true;
            std::vector<std::string> errors;
            
            if (result.status != core::RunStatus::Success) {
                passed = false;
                errors.push_back("Optimization failed");
            }
            
            if (result.best_fitness < ptest.min_fitness || 
                result.best_fitness > ptest.max_fitness) {
                passed = false;
                errors.push_back("Fitness out of expected range [" + 
                               std::to_string(ptest.min_fitness) + ", " +
                               std::to_string(ptest.max_fitness) + "]");
            }
            
            if (result.best_solution.size() != ptest.problem->dimension()) {
                passed = false;
                errors.push_back("Solution dimension mismatch");
            }
            
            // Check bounds
            auto lower = ptest.problem->lower_bounds();
            auto upper = ptest.problem->upper_bounds();
            if (!verify_solution_bounds(result.best_solution, lower, upper)) {
                passed = false;
                errors.push_back("Solution violates bounds");
            }
            
            if (passed) {
                std::cout << "  " << ptest.name << ": PASSED (fitness = " 
                         << result.best_fitness << ")\n";
                problem_tests_passed++;
            } else {
                std::cout << "  " << ptest.name << ": FAILED\n";
                for (const auto &err : errors) {
                    std::cout << "      - " << err << "\n";
                }
                problem_tests_failed++;
            }
        }
        
        if (problem_tests_failed == 0) {
            tests_passed++;
        } else {
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
        std::cout << "\n  All correctness tests PASSED!\n";
        return 0;
    } else {
        std::cout << "\n  Some tests FAILED!\n";
        return 1;
    }
}

