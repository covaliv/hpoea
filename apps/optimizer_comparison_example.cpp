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
#include <cmath>
#include <numeric>

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
        config.max_parallel_trials = 1;
        config.algorithm_budget.generations = 30;
        config.optimizer_budget = budget;
        config.log_file_path = "comparison_" + opt_config.name + ".jsonl";

        core::JsonlLogger logger(config.log_file_path);
        core::SequentialExperimentManager manager;

        auto result = manager.run_experiment(config, *opt_config.optimizer, ea_factory, problem, logger);

        std::vector<double> objectives;
        std::size_t excluded = 0;
        for (const auto &r : result.optimizer_results) {
            const bool success_like = r.status == core::RunStatus::Success
                || r.status == core::RunStatus::BudgetExceeded;
            if (success_like && std::isfinite(r.best_objective)) {
                objectives.push_back(r.best_objective);
            } else {
                excluded += 1;
            }
        }

        if (!objectives.empty()) {
            const double sum = std::accumulate(objectives.begin(), objectives.end(), 0.0);
            const double best = *std::min_element(objectives.begin(), objectives.end());
            const double avg = sum / static_cast<double>(objectives.size());
            results.push_back({opt_config.name, best});

            std::cout << std::fixed << std::setprecision(6);
            std::cout << opt_config.name << ": best=" << best << " avg=" << avg
                      << " (included: " << objectives.size()
                      << ", excluded: " << excluded << ")\n";
        } else {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << opt_config.name << ": no successful trials"
                      << " (excluded: " << excluded << ")\n";
        }
    }

    if (!results.empty()) {
        std::sort(results.begin(), results.end(),
            [](const auto &a, const auto &b) { return a.second < b.second; });

        std::cout << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < results.size(); ++i) {
            std::cout << (i + 1) << ". " << results[i].first
                      << ": best=" << results[i].second << "\n";
        }
    }

    return 0;
}

