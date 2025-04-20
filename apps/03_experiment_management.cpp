#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <iostream>
#include <iomanip>

int main() {
    using namespace hpoea;
    
    wrappers::problems::AckleyProblem problem(10);
    pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
    pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
    
    core::ParameterSet optimizer_params;
    optimizer_params.emplace("generations", static_cast<std::int64_t>(15));
    optimizer.configure(optimizer_params);
    
    core::ExperimentConfig config;
    config.experiment_id = "advanced_example";
    config.trials_per_optimizer = 5;
    config.islands = 2;
    config.algorithm_budget.generations = 50;
    config.optimizer_budget.generations = 15;
    config.optimizer_budget.function_evaluations = 3000;
    config.log_file_path = "experiment_results.jsonl";
    
    core::JsonlLogger logger(config.log_file_path);
    core::SequentialExperimentManager manager;
    
    auto result = manager.run_experiment(config, optimizer, ea_factory, problem, logger);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "experiment_id: " << result.experiment_id << "\n";
    std::cout << "optimizer_runs: " << result.optimizer_results.size() << "\n";
    
    if (!result.optimizer_results.empty()) {
        const auto &best_result = result.optimizer_results[0];
        std::cout << "best_objective: " << best_result.best_objective << "\n";
        std::cout << "trials: " << best_result.trials.size() << "\n";
        std::cout << "function_evaluations: " << best_result.budget_usage.function_evaluations << "\n";
        
        if (!best_result.best_parameters.empty()) {
            for (const auto &[name, value] : best_result.best_parameters) {
                std::cout << name << ": ";
                std::visit([](auto v) { std::cout << v; }, value);
                std::cout << "\n";
            }
        }
    }
    
    std::cout << "log_file: " << config.log_file_path << "\n";
    
    return 0;
}

