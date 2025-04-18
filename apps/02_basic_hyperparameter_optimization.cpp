#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/types.hpp"

#include <iostream>
#include <iomanip>

int main() {
    using namespace hpoea;
    
    wrappers::problems::RosenbrockProblem problem(8);
    pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
    pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
    
    core::ParameterSet optimizer_params;
    optimizer_params.emplace("generations", static_cast<std::int64_t>(20));
    optimizer_params.emplace("sigma0", 0.3);
    optimizer.configure(optimizer_params);
    
    core::Budget budget;
    budget.generations = 20;
    budget.function_evaluations = 10000;
    
    auto result = optimizer.optimize(ea_factory, problem, budget, 42UL);
    
    if (result.status == core::RunStatus::Success) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "best_objective: " << result.best_objective << "\n";
        std::cout << "trials: " << result.trials.size() << "\n";
        
        for (const auto &[name, value] : result.best_parameters) {
            std::cout << name << ": ";
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

