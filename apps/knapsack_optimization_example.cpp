#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/types.hpp"

#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    using namespace hpoea;
    
    std::vector<double> values = {135.0, 139.0, 149.0, 150.0, 156.0, 163.0, 173.0, 184.0, 192.0, 201.0, 210.0, 214.0, 221.0, 229.0, 240.0};
    std::vector<double> weights = {70.0, 73.0, 77.0, 80.0, 82.0, 87.0, 90.0, 94.0, 98.0, 106.0, 110.0, 113.0, 115.0, 118.0, 120.0};
    double capacity = 750.0;
    
    wrappers::problems::KnapsackProblem problem(values, weights, capacity);
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    auto algorithm = factory.create();
    
    core::ParameterSet params;
    params.emplace("population_size", static_cast<std::int64_t>(50));
    params.emplace("generations", static_cast<std::int64_t>(200));
    params.emplace("scaling_factor", 0.8);
    params.emplace("crossover_rate", 0.9);
    algorithm->configure(params);
    
    core::Budget budget;
    budget.generations = 200;
    
    auto result = algorithm->run(problem, budget, 42UL);
    
    if (result.status == core::RunStatus::Success) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "best_fitness: " << result.best_fitness << "\n";
        
        double total_value = 0.0;
        double total_weight = 0.0;
        std::vector<std::size_t> selected_items;
        
        for (std::size_t i = 0; i < result.best_solution.size(); ++i) {
            if (result.best_solution[i] >= 0.5) {
                selected_items.push_back(i);
                total_value += values[i];
                total_weight += weights[i];
            }
        }
        
        std::cout << "selected_items: [";
        for (std::size_t i = 0; i < selected_items.size(); ++i) {
            std::cout << selected_items[i];
            if (i < selected_items.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        
        std::cout << "total_value: " << total_value << "\n";
        std::cout << "total_weight: " << total_weight << "\n";
        std::cout << "capacity: " << capacity << "\n";
        std::cout << "function_evaluations: " << result.budget_usage.function_evaluations << "\n";
        std::cout << "generations: " << result.budget_usage.generations << "\n";
        std::cout << "wall_time_ms: " << result.budget_usage.wall_time.count() << "\n";
    } else {
        std::cerr << "error: " << result.message << "\n";
        return 1;
    }
    
    return 0;
}

