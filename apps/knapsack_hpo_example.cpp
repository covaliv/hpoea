#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    using namespace hpoea;
    
    std::vector<double> values = {135.0, 139.0, 149.0, 150.0, 156.0, 163.0, 173.0, 184.0, 192.0, 201.0, 210.0, 214.0, 221.0, 229.0, 240.0};
    std::vector<double> weights = {70.0, 73.0, 77.0, 80.0, 82.0, 87.0, 90.0, 94.0, 98.0, 106.0, 110.0, 113.0, 115.0, 118.0, 120.0};
    double capacity = 750.0;
    
    wrappers::problems::KnapsackProblem problem(values, weights, capacity);
    pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
    pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
    
    core::ParameterSet optimizer_params;
    optimizer_params.emplace("generations", static_cast<std::int64_t>(20));
    optimizer_params.emplace("sigma0", 0.3);
    optimizer.configure(optimizer_params);
    
    core::Budget budget;
    budget.generations = 20;
    budget.function_evaluations = 5000;
    
    auto result = optimizer.optimize(ea_factory, problem, budget, 42UL);
    
    if (result.status == core::RunStatus::Success) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "best_objective: " << result.best_objective << "\n";
        std::cout << "trials: " << result.trials.size() << "\n";
        
        std::cout << "best_hyperparameters:\n";
        for (const auto &[name, value] : result.best_parameters) {
            std::cout << "  " << name << ": ";
            std::visit([](auto v) { std::cout << v; }, value);
            std::cout << "\n";
        }
        
        std::cout << "function_evaluations: " << result.budget_usage.function_evaluations << "\n";
        std::cout << "wall_time_ms: " << result.budget_usage.wall_time.count() << "\n";
    } else {
        std::cerr << "error: " << result.message << "\n";
        return 1;
    }
    
    return 0;
}

