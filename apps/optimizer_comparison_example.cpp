#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

int main() {
    using namespace hpoea;
    
    wrappers::problems::RastriginProblem problem(12);
    pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;
    
    struct OptimizerConfig {
        std::string name;
        std::unique_ptr<core::IHyperparameterOptimizer> optimizer;
        core::ParameterSet params;
    };
    
    std::vector<OptimizerConfig> optimizers;
    
    {
        auto opt = std::make_unique<pagmo_wrappers::PagmoCmaesHyperOptimizer>();
        core::ParameterSet params;
        params.emplace("generations", static_cast<std::int64_t>(20));
        params.emplace("sigma0", 0.3);
        opt->configure(params);
        optimizers.push_back({"CMA-ES", std::move(opt), std::move(params)});
    }
    
    {
        auto opt = std::make_unique<pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer>();
        core::ParameterSet params;
        params.emplace("iterations", static_cast<std::int64_t>(50));
        params.emplace("ts", 100.0);
        params.emplace("tf", 0.01);
        opt->configure(params);
        optimizers.push_back({"SimulatedAnnealing", std::move(opt), std::move(params)});
    }
    
    {
        auto opt = std::make_unique<pagmo_wrappers::PagmoPsoHyperOptimizer>();
        core::ParameterSet params;
        params.emplace("generations", static_cast<std::int64_t>(30));
        params.emplace("omega", 0.7298);
        params.emplace("eta1", 2.05);
        params.emplace("eta2", 2.05);
        opt->configure(params);
        optimizers.push_back({"PSO", std::move(opt), std::move(params)});
    }
    
    core::Budget budget;
    budget.generations = 15;
    budget.function_evaluations = 3000;
    
    std::vector<std::pair<std::string, double>> results;
    
    for (auto &opt_config : optimizers) {
        core::ExperimentConfig config;
        config.experiment_id = "comparison_" + opt_config.name;
        config.trials_per_optimizer = 3;
        config.islands = 1;
        config.algorithm_budget.generations = 30;
        config.optimizer_budget = budget;
        config.log_file_path = "comparison_" + opt_config.name + ".jsonl";
        
        core::JsonlLogger logger(config.log_file_path);
        core::SequentialExperimentManager manager;
        
        auto result = manager.run_experiment(config, *opt_config.optimizer, ea_factory, problem, logger);
        
        if (!result.optimizer_results.empty()) {
            const auto &best = result.optimizer_results[0];
            results.push_back({opt_config.name, best.best_objective});
            
            std::cout << std::fixed << std::setprecision(6);
            std::cout << opt_config.name << ": " << best.best_objective 
                      << " (trials: " << best.trials.size() 
                      << ", evals: " << best.budget_usage.function_evaluations << ")\n";
        }
    }
    
    if (!results.empty()) {
        std::sort(results.begin(), results.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });
        
        std::cout << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < results.size(); ++i) {
            std::cout << (i + 1) << ". " << results[i].first 
                      << ": " << results[i].second << "\n";
        }
    }
    
    return 0;
}

