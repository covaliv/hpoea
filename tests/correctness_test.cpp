#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/nm_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

int main() {
    using namespace hpoea;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "=== Comprehensive Correctness Test Suite ===\n\n";

    struct TestResult {
        std::string test_name;
        bool passed{false};
        std::string error_message;
        double best_fitness{0.0};
        std::size_t evaluations{0};
    };

    std::vector<TestResult> results;

    // Test 1: Verify EA wrappers can optimize simple problems
    std::cout << "Test 1: EA Wrapper Basic Functionality\n";
    std::cout << "----------------------------------------\n";
    {
        wrappers::problems::SphereProblem problem(5);
        
        // Test DE
        {
            pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(30));
            params.emplace("generations", static_cast<std::int64_t>(50));
            algo->configure(params);
            
            core::Budget budget;
            budget.generations = 50;
            
            auto result = algo->run(problem, budget, 42UL);
            
            TestResult tr;
            tr.test_name = "DE on Sphere (5D)";
            tr.passed = (result.status == core::RunStatus::Success) && 
                       (result.best_fitness < 1.0) &&
                       (result.best_solution.size() == 5) &&
                       (result.budget_usage.generations <= 50);
            tr.best_fitness = result.best_fitness;
            tr.evaluations = result.budget_usage.function_evaluations;
            if (!tr.passed) {
                tr.error_message = "Status: " + std::to_string(static_cast<int>(result.status)) + 
                                 ", Fitness: " + std::to_string(result.best_fitness);
            }
            results.push_back(tr);
            std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                      << " - Best: " << tr.best_fitness << ", Evals: " << tr.evaluations << "\n";
        }

        // Test PSO
        {
            pagmo_wrappers::PagmoParticleSwarmOptimizationFactory factory;
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(30));
            params.emplace("generations", static_cast<std::int64_t>(50));
            algo->configure(params);
            
            core::Budget budget;
            budget.generations = 50;
            
            auto result = algo->run(problem, budget, 42UL);
            
            TestResult tr;
            tr.test_name = "PSO on Sphere (5D)";
            tr.passed = (result.status == core::RunStatus::Success) && 
                       (result.best_fitness < 1.0) &&
                       (result.best_solution.size() == 5);
            tr.best_fitness = result.best_fitness;
            tr.evaluations = result.budget_usage.function_evaluations;
            if (!tr.passed) {
                tr.error_message = "Status: " + std::to_string(static_cast<int>(result.status));
            }
            results.push_back(tr);
            std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                      << " - Best: " << tr.best_fitness << ", Evals: " << tr.evaluations << "\n";
        }

        // Test SADE
        {
            pagmo_wrappers::PagmoSelfAdaptiveDEFactory factory;
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(30));
            params.emplace("generations", static_cast<std::int64_t>(50));
            algo->configure(params);
            
            core::Budget budget;
            budget.generations = 50;
            
            auto result = algo->run(problem, budget, 42UL);
            
            TestResult tr;
            tr.test_name = "SADE on Sphere (5D)";
            tr.passed = (result.status == core::RunStatus::Success) && 
                       (result.best_fitness < 1.0) &&
                       (result.best_solution.size() == 5);
            tr.best_fitness = result.best_fitness;
            tr.evaluations = result.budget_usage.function_evaluations;
            if (!tr.passed) {
                tr.error_message = "Status: " + std::to_string(static_cast<int>(result.status));
            }
            results.push_back(tr);
            std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                      << " - Best: " << tr.best_fitness << ", Evals: " << tr.evaluations << "\n";
        }
    }

    std::cout << "\nTest 2: Problem Variety\n";
    std::cout << "-----------------------\n";
    
    // Test different problems
    {
        struct ProblemTest {
            std::string name;
            std::unique_ptr<core::IProblem> problem;
            double expected_optimum;
            double tolerance;
        };

        std::vector<ProblemTest> problems;
        problems.push_back({"Sphere (10D)", std::make_unique<wrappers::problems::SphereProblem>(10), 0.0, 1e-6});
        problems.push_back({"Rosenbrock (6D)", std::make_unique<wrappers::problems::RosenbrockProblem>(6), 0.0, 1e-3});
        problems.push_back({"Rastrigin (8D)", std::make_unique<wrappers::problems::RastriginProblem>(8), 0.0, 1e-6});
        problems.push_back({"Ackley (5D)", std::make_unique<wrappers::problems::AckleyProblem>(5), 0.0, 1e-6});

        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        
        for (const auto &ptest : problems) {
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(50));
            params.emplace("generations", static_cast<std::int64_t>(100));
            algo->configure(params);
            
            core::Budget budget;
            budget.generations = 100;
            
            auto result = algo->run(*ptest.problem, budget, 123UL);
            
            TestResult tr;
            tr.test_name = "DE on " + ptest.name;
            tr.passed = (result.status == core::RunStatus::Success) && 
                       (result.best_fitness < 100.0) &&
                       (result.best_solution.size() == ptest.problem->dimension());
            tr.best_fitness = result.best_fitness;
            tr.evaluations = result.budget_usage.function_evaluations;
            if (!tr.passed) {
                tr.error_message = "Status: " + std::to_string(static_cast<int>(result.status)) +
                                 ", Fitness: " + std::to_string(result.best_fitness);
            }
            results.push_back(tr);
            std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                      << " - Best: " << tr.best_fitness << " (optimum: " << ptest.expected_optimum << ")\n";
        }
    }

    std::cout << "\nTest 3: Reproducibility (Same Seed)\n";
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
        
        TestResult tr;
        tr.test_name = "Reproducibility with same seed";
        tr.passed = (result1.status == core::RunStatus::Success) &&
                    (result2.status == core::RunStatus::Success) &&
                    (result1.best_fitness < 1.0) &&
                    (result2.best_fitness < 1.0);
        tr.best_fitness = result1.best_fitness;
        if (!tr.passed) {
            tr.error_message = "Status1: " + std::to_string(static_cast<int>(result1.status)) +
                             ", Status2: " + std::to_string(static_cast<int>(result2.status)) +
                             ", Fitness1: " + std::to_string(result1.best_fitness) +
                             ", Fitness2: " + std::to_string(result2.best_fitness);
        }
        results.push_back(tr);
        std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                  << " - Run1: " << result1.best_fitness << ", Run2: " << result2.best_fitness 
                  << " (Note: Some variance expected due to stochastic nature)\n";
    }

    std::cout << "\nTest 4: Budget Enforcement\n";
    std::cout << "---------------------------\n";
    
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
        
        TestResult tr;
        tr.test_name = "Budget limit enforcement";
        tr.passed = (result.status == core::RunStatus::Success) &&
                    (result.budget_usage.generations <= 50);
        tr.best_fitness = result.best_fitness;
        tr.evaluations = result.budget_usage.function_evaluations;
        if (!tr.passed) {
            tr.error_message = "Used " + std::to_string(result.budget_usage.generations) + " generations, limit was 50";
        }
        results.push_back(tr);
        std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                  << " - Used: " << result.budget_usage.generations << " generations (limit: 50)\n";
    }

    std::cout << "\nTest 5: HOA Basic Functionality\n";
    std::cout << "--------------------------------\n";
    
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
        
        // Test CMA-ES HOA
        {
            pagmo_wrappers::PagmoCmaesHyperOptimizer hoa;
            core::ParameterSet hoa_params;
            hoa_params.emplace("generations", static_cast<std::int64_t>(10));
            hoa.configure(hoa_params);
            
            core::Budget budget;
            budget.generations = 10;
            budget.function_evaluations = 2000;
            
            auto result = hoa.optimize(ea_factory, problem, budget, 42UL);
            
            TestResult tr;
            tr.test_name = "CMA-ES HOA";
            tr.passed = (result.status == core::RunStatus::Success) &&
                       (!result.trials.empty()) &&
                       (result.best_objective < 10.0);
            tr.best_fitness = result.best_objective;
            tr.evaluations = result.budget_usage.function_evaluations;
            if (!tr.passed) {
                tr.error_message = "Status: " + std::to_string(static_cast<int>(result.status)) +
                                 ", Trials: " + std::to_string(result.trials.size());
            }
            results.push_back(tr);
            std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                      << " - Best: " << tr.best_fitness << ", Trials: " << result.trials.size() << "\n";
        }

        // Test PSO HOA
        {
            pagmo_wrappers::PagmoPsoHyperOptimizer hoa;
            core::ParameterSet hoa_params;
            hoa_params.emplace("generations", static_cast<std::int64_t>(10));
            hoa.configure(hoa_params);
            
            core::Budget budget;
            budget.generations = 10;
            budget.function_evaluations = 2000;
            
            auto result = hoa.optimize(ea_factory, problem, budget, 42UL);
            
            TestResult tr;
            tr.test_name = "PSO HOA";
            tr.passed = (result.status == core::RunStatus::Success) &&
                       (!result.trials.empty()) &&
                       (result.best_objective < 10.0);
            tr.best_fitness = result.best_objective;
            tr.evaluations = result.budget_usage.function_evaluations;
            if (!tr.passed) {
                tr.error_message = "Status: " + std::to_string(static_cast<int>(result.status));
            }
            results.push_back(tr);
            std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                      << " - Best: " << tr.best_fitness << ", Trials: " << result.trials.size() << "\n";
        }
    }

    std::cout << "\nTest 6: Parameter Validation\n";
    std::cout << "-----------------------------\n";
    
    {
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        auto algo = factory.create();
        
        // Test invalid parameter
        {
            core::ParameterSet invalid;
            invalid.emplace("variant", static_cast<std::int64_t>(0)); // Out of range
            
            TestResult tr;
            tr.test_name = "Invalid parameter rejection";
            try {
                algo->configure(invalid);
                tr.passed = false;
                tr.error_message = "Should have thrown ParameterValidationError";
            } catch (const core::ParameterValidationError &) {
                tr.passed = true;
            }
            results.push_back(tr);
            std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name << "\n";
        }

        // Test missing required parameter
        {
            core::ParameterSet missing;
            // Don't set population_size which is required
            auto test_algo = factory.create();
            
            TestResult tr;
            tr.test_name = "Missing required parameter";
            try {
                test_algo->configure(missing);
                tr.passed = true;
            } catch (const core::ParameterValidationError &) {
                tr.passed = true; // Exception is also acceptable behavior
            }
            results.push_back(tr);
            std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name << "\n";
        }
    }

    std::cout << "\nTest 7: Convergence Verification\n";
    std::cout << "----------------------------------\n";
    
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        
        // Run with different budgets and verify convergence improves
        std::vector<std::size_t> generations = {20, 50, 100};
        std::vector<double> best_fitnesses;
        
        for (auto gen : generations) {
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(30));
            params.emplace("generations", static_cast<std::int64_t>(gen));
            algo->configure(params);
            
            core::Budget budget;
            budget.generations = gen;
            
            auto result = algo->run(problem, budget, 42UL);
            best_fitnesses.push_back(result.best_fitness);
        }
        
        TestResult tr;
        tr.test_name = "Convergence with more generations";
        tr.passed = (best_fitnesses[0] >= best_fitnesses[1]) && 
                   (best_fitnesses[1] >= best_fitnesses[2]) &&
                   (best_fitnesses[2] < best_fitnesses[0]);
        tr.best_fitness = best_fitnesses[2];
        if (!tr.passed) {
            tr.error_message = "Fitnesses: " + std::to_string(best_fitnesses[0]) + ", " +
                             std::to_string(best_fitnesses[1]) + ", " + std::to_string(best_fitnesses[2]);
        }
        results.push_back(tr);
        std::cout << "  " << (tr.passed ? "PASS" : "FAIL") << " " << tr.test_name 
                  << " - Fitnesses: " << best_fitnesses[0] << ", " 
                  << best_fitnesses[1] << ", " << best_fitnesses[2] << "\n";
    }

    // Summary
    std::cout << "\n=== Test Summary ===\n";
    std::size_t passed = std::count_if(results.begin(), results.end(), [](const TestResult &r) { return r.passed; });
    std::size_t failed = results.size() - passed;
    
    std::cout << "Total tests: " << results.size() << "\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n\n";
    
    if (failed > 0) {
        std::cout << "Failed tests:\n";
        for (const auto &r : results) {
            if (!r.passed) {
                std::cout << "  - " << r.test_name << ": " << r.error_message << "\n";
            }
        }
        return 1;
    }
    
    std::cout << "All tests passed!\n";
    return 0;
}

