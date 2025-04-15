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

int main() {
    using namespace hpoea;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "========================================\n";
    std::cout << "  HPOEA Framework Example Program\n";
    std::cout << "========================================\n\n";
    
    // Test 1: Simple EA optimization
    std::cout << "Test 1: Optimizing Sphere Problem (5D) with DE\n";
    std::cout << "------------------------------------------------\n";
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
        
        std::cout << "  Configuration:\n";
        std::cout << "    Population size: 30\n";
        std::cout << "    Generations: 50\n";
        std::cout << "    DE variant: 2\n";
        std::cout << "    Scaling factor (F): 0.8\n";
        std::cout << "    Crossover rate (CR): 0.9\n";
        std::cout << "  Running optimization...\n";
        
        auto result = algorithm->run(problem, budget, 42UL);
        
        if (result.status == core::RunStatus::Success) {
            std::cout << "  Optimization completed successfully!\n";
            std::cout << "  Best fitness: " << result.best_fitness << "\n";
            std::cout << "  Best solution: [";
            for (size_t i = 0; i < result.best_solution.size(); ++i) {
                std::cout << result.best_solution[i];
                if (i < result.best_solution.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
            std::cout << "  Function evaluations: " << result.budget_usage.function_evaluations << "\n";
            std::cout << "  Generations used: " << result.budget_usage.generations << "\n";
            std::cout << "  Wall time: " << result.budget_usage.wall_time.count() << " ms\n";
            
            // Verify correctness
            bool correct = (result.best_fitness < 1.0) && 
                          (result.best_solution.size() == 5) &&
                          (result.budget_usage.generations <= 50);
            std::cout << "  Correctness check: " << (correct ? "PASSED" : "FAILED") << "\n";
        } else {
            std::cout << "  Optimization failed: " << result.message << "\n";
        }
    }
    
    std::cout << "\n";
    
    // Test 2: Compare multiple algorithms
    std::cout << "Test 2: Comparing Multiple Algorithms on Sphere (10D)\n";
    std::cout << "-----------------------------------------------------\n";
    {
        wrappers::problems::SphereProblem problem(10);
        
        struct AlgorithmTest {
            std::string name;
            std::unique_ptr<core::IEvolutionaryAlgorithmFactory> factory;
        };
        
        std::vector<AlgorithmTest> algorithms;
        algorithms.push_back({"DE", std::make_unique<pagmo_wrappers::PagmoDifferentialEvolutionFactory>()});
        algorithms.push_back({"PSO", std::make_unique<pagmo_wrappers::PagmoParticleSwarmOptimizationFactory>()});
        algorithms.push_back({"SADE", std::make_unique<pagmo_wrappers::PagmoSelfAdaptiveDEFactory>()});
        
        core::Budget budget;
        budget.generations = 100;
        
        std::vector<std::pair<std::string, double>> results;
        
        for (const auto &alg_test : algorithms) {
            auto algo = alg_test.factory->create();
            
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(50));
            params.emplace("generations", static_cast<std::int64_t>(100));
            
            if (alg_test.name == "PSO") {
                params.emplace("omega", 0.7298);
                params.emplace("eta1", 2.05);
                params.emplace("eta2", 2.05);
                params.emplace("max_velocity", 0.5);
                params.emplace("variant", static_cast<std::int64_t>(5));
            }
            
            algo->configure(params);
            
            std::cout << "  Running " << alg_test.name << "... ";
            auto result = algo->run(problem, budget, 42UL);
            
            if (result.status == core::RunStatus::Success) {
                results.push_back({alg_test.name, result.best_fitness});
                std::cout << "Best fitness: " << result.best_fitness 
                          << " (evals: " << result.budget_usage.function_evaluations << ")\n";
            } else {
                std::cout << "FAILED\n";
            }
        }
        
        // Find best algorithm
        if (!results.empty()) {
            auto best = std::min_element(results.begin(), results.end(),
                [](const auto &a, const auto &b) { return a.second < b.second; });
            std::cout << "\n  Best algorithm: " << best->first 
                      << " with fitness " << best->second << "\n";
        }
    }
    
    std::cout << "\n";
    
    // Test 3: Hyperparameter optimization
    std::cout << "Test 3: Hyperparameter Optimization (CMA-ES tuning DE)\n";
    std::cout << "------------------------------------------------------\n";
    {
        wrappers::problems::SphereProblem problem(5);
        pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
        pagmo_wrappers::PagmoCmaesHyperOptimizer hoa;
        
        core::ParameterSet hoa_params;
        hoa_params.emplace("generations", static_cast<std::int64_t>(15));
        hoa_params.emplace("sigma0", 0.5);
        hoa.configure(hoa_params);
        
        core::Budget budget;
        budget.generations = 15;
        budget.function_evaluations = 5000;
        
        std::cout << "  HOA Configuration:\n";
        std::cout << "    Algorithm: CMA-ES\n";
        std::cout << "    Generations: 15\n";
        std::cout << "    Initial sigma: 0.5\n";
        std::cout << "    Budget: " << budget.function_evaluations.value() << " evaluations\n";
        std::cout << "  Running hyperparameter optimization...\n";
        
        auto result = hoa.optimize(ea_factory, problem, budget, 42UL);
        
        if (result.status == core::RunStatus::Success) {
            std::cout << "  Hyperparameter optimization completed!\n";
            std::cout << "  Best objective: " << result.best_objective << "\n";
            std::cout << "  Number of trials: " << result.trials.size() << "\n";
            std::cout << "  Best hyperparameters found:\n";
            for (const auto &[name, value] : result.best_parameters) {
                std::cout << "    " << name << " = ";
                std::visit([](auto v) { std::cout << v; }, value);
                std::cout << "\n";
            }
            std::cout << "  Function evaluations: " << result.budget_usage.function_evaluations << "\n";
            std::cout << "  Wall time: " << result.budget_usage.wall_time.count() << " ms\n";
            
            // Show top 3 trials
            std::vector<core::HyperparameterTrialRecord> sorted_trials = result.trials;
            std::sort(sorted_trials.begin(), sorted_trials.end(),
                [](const auto &a, const auto &b) {
                    return a.optimization_result.best_fitness < b.optimization_result.best_fitness;
                });
            
            std::cout << "\n  Top 3 configurations:\n";
            for (size_t i = 0; i < std::min(3UL, sorted_trials.size()); ++i) {
                std::cout << "    Rank " << (i+1) << ": fitness = " 
                          << sorted_trials[i].optimization_result.best_fitness << "\n";
            }
            
            // Verify correctness
            bool correct = (!result.trials.empty()) &&
                          (std::isfinite(result.best_objective)) &&
                          (result.best_objective < 10.0);
            std::cout << "  Correctness check: " << (correct ? "PASSED" : "FAILED") << "\n";
        } else {
            std::cout << "  Hyperparameter optimization failed: " << result.message << "\n";
        }
    }
    
    std::cout << "\n";
    
    // Test 4: Multiple problems
    std::cout << "Test 4: Testing DE on Multiple Benchmark Problems\n";
    std::cout << "--------------------------------------------------\n";
    {
        struct ProblemTest {
            std::string name;
            std::unique_ptr<core::IProblem> problem;
            double optimum;
        };
        
        std::vector<ProblemTest> problems;
        problems.push_back({"Sphere (5D)", std::make_unique<wrappers::problems::SphereProblem>(5), 0.0});
        problems.push_back({"Rosenbrock (6D)", std::make_unique<wrappers::problems::RosenbrockProblem>(6), 0.0});
        problems.push_back({"Rastrigin (8D)", std::make_unique<wrappers::problems::RastriginProblem>(8), 0.0});
        problems.push_back({"Ackley (5D)", std::make_unique<wrappers::problems::AckleyProblem>(5), 0.0});
        
        pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
        
        core::Budget budget;
        budget.generations = 100;
        
        for (const auto &ptest : problems) {
            auto algo = factory.create();
            core::ParameterSet params;
            params.emplace("population_size", static_cast<std::int64_t>(50));
            params.emplace("generations", static_cast<std::int64_t>(100));
            algo->configure(params);
            
            std::cout << "  " << ptest.name << ": ";
            auto result = algo->run(*ptest.problem, budget, 42UL);
            
            if (result.status == core::RunStatus::Success) {
                double distance_to_optimum = std::abs(result.best_fitness - ptest.optimum);
                std::cout << "fitness = " << result.best_fitness 
                          << " (distance to optimum: " << distance_to_optimum << ")\n";
                
                // Verify correctness
                bool correct = (result.best_fitness < 100.0) &&
                              (result.best_solution.size() == ptest.problem->dimension());
                if (!correct) {
                    std::cout << "    Correctness check FAILED\n";
                }
            } else {
                std::cout << "FAILED\n";
            }
        }
    }
    
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  All Tests Completed!\n";
    std::cout << "========================================\n";
    
    return 0;
}

