#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

int main() {
    using namespace hpoea;

    const bool verbose = true; // Always verbose for benchmarks

    struct BenchmarkConfig {
        std::string name;
        std::unique_ptr<core::IProblem> problem;
        std::unique_ptr<core::IEvolutionaryAlgorithmFactory> ea_factory;
        std::unique_ptr<core::IHyperparameterOptimizer> hoa;
        core::ParameterSet hoa_params;
        core::Budget algorithm_budget;
        core::Budget optimizer_budget;
        std::size_t trials;
    };

    std::vector<BenchmarkConfig> benchmarks;

    // Benchmark 1: DE + CMA-ES on Sphere
    {
        BenchmarkConfig config;
        config.name = "DE_CMAES_Sphere";
        config.problem = std::make_unique<wrappers::problems::SphereProblem>(10);
        config.ea_factory = std::make_unique<pagmo_wrappers::PagmoDifferentialEvolutionFactory>();
        config.hoa = std::make_unique<pagmo_wrappers::PagmoCmaesHyperOptimizer>();
        config.hoa_params.emplace("generations", static_cast<std::int64_t>(30));
        config.algorithm_budget.generations = 100;
        config.optimizer_budget.generations = 30;
        config.optimizer_budget.function_evaluations = 5000;
        config.trials = 3;
        benchmarks.push_back(std::move(config));
    }

    // Benchmark 2: PSO + PSO-Hyper on Rastrigin
    {
        BenchmarkConfig config;
        config.name = "PSO_PSOHyper_Rastrigin";
        config.problem = std::make_unique<wrappers::problems::RastriginProblem>(8);
        config.ea_factory = std::make_unique<pagmo_wrappers::PagmoParticleSwarmOptimizationFactory>();
        config.hoa = std::make_unique<pagmo_wrappers::PagmoPsoHyperOptimizer>();
        config.hoa_params.emplace("generations", static_cast<std::int64_t>(25));
        config.algorithm_budget.generations = 80;
        config.optimizer_budget.generations = 25;
        config.optimizer_budget.function_evaluations = 4000;
        config.trials = 3;
        benchmarks.push_back(std::move(config));
    }

    // Benchmark 3: SADE + SA on Rosenbrock
    {
        BenchmarkConfig config;
        config.name = "SADE_SA_Rosenbrock";
        config.problem = std::make_unique<wrappers::problems::RosenbrockProblem>(6);
        config.ea_factory = std::make_unique<pagmo_wrappers::PagmoSelfAdaptiveDEFactory>();
        config.hoa = std::make_unique<pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer>();
        config.hoa_params.emplace("iterations", static_cast<std::int64_t>(40));
        config.algorithm_budget.generations = 120;
        config.optimizer_budget.function_evaluations = 3000;
        config.trials = 3;
        benchmarks.push_back(std::move(config));
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Hyperparameter Optimization Benchmark Suite ===\n\n";

    for (auto &benchmark : benchmarks) {
        std::cout << "Benchmark: " << benchmark.name << '\n';
        std::cout << "  Problem: " << benchmark.problem->metadata().id
                  << " (dim=" << benchmark.problem->dimension() << ")\n";
        std::cout << "  EA: " << benchmark.ea_factory->identity().family << '\n';
        std::cout << "  HOA: " << benchmark.hoa->identity().family << '\n';
        std::cout << "  Trials: " << benchmark.trials << '\n';

        benchmark.hoa->configure(benchmark.hoa_params);

        core::ExperimentConfig exp_config;
        exp_config.experiment_id = benchmark.name;
        exp_config.trials_per_optimizer = benchmark.trials;
        exp_config.islands = 1;
        exp_config.algorithm_budget = benchmark.algorithm_budget;
        exp_config.optimizer_budget = benchmark.optimizer_budget;
        exp_config.log_file_path = benchmark.name + "_benchmark.jsonl";

        if (std::filesystem::exists(exp_config.log_file_path)) {
            std::filesystem::remove(exp_config.log_file_path);
        }

        core::JsonlLogger logger(exp_config.log_file_path);
        core::SequentialExperimentManager manager;

        const auto start_time = std::chrono::steady_clock::now();
        auto result = manager.run_experiment(exp_config, *benchmark.hoa, *benchmark.ea_factory, *benchmark.problem, logger);
        const auto end_time = std::chrono::steady_clock::now();

        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::vector<double> best_objectives;
        std::size_t total_trials = 0;
        std::size_t total_fevals = 0;

        for (const auto &opt_result : result.optimizer_results) {
            best_objectives.push_back(opt_result.best_objective);
            total_trials += opt_result.trials.size();
            total_fevals += opt_result.budget_usage.function_evaluations;
        }

        const double avg_best = std::accumulate(best_objectives.begin(), best_objectives.end(), 0.0) / best_objectives.size();
        const double min_best = *std::min_element(best_objectives.begin(), best_objectives.end());
        const double max_best = *std::max_element(best_objectives.begin(), best_objectives.end());

        std::cout << "  Results:\n";
        std::cout << "    Wall time: " << duration.count() << " ms\n";
        std::cout << "    Total trials: " << total_trials << '\n';
        std::cout << "    Total function evaluations: " << total_fevals << '\n';
        std::cout << "    Best objective (min/avg/max): " << min_best << " / " << avg_best << " / " << max_best << '\n';
        std::cout << '\n';
    }

    std::cout << "=== Benchmark Suite Complete ===\n";

    return 0;
}

