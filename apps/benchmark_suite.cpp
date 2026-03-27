#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace hpoea;

namespace {

struct Benchmark {
    std::string name;
    std::unique_ptr<core::IProblem> problem;
    std::unique_ptr<core::IEvolutionaryAlgorithmFactory> ea;
    std::unique_ptr<core::IHyperparameterOptimizer> hoa;
    core::ParameterSet hoa_params;
    core::Budget algo_budget;
    core::Budget opt_budget;
    std::size_t trials;
};

struct Stats {
    double min, avg, max;
};

Stats compute_stats(const std::vector<double> &v) {
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return {*std::min_element(v.begin(), v.end()),
            sum / static_cast<double>(v.size()),
            *std::max_element(v.begin(), v.end())};
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "hpoea benchmark suite\n\n";

    const bool full_mode = [] {
        const char *value = std::getenv("HPOEA_BENCHMARK_FULL");
        return value != nullptr && std::string(value) == "1";
    }();

    std::cout << "mode: " << (full_mode ? "full" : "fast") << "\n\n";

    std::vector<Benchmark> benchmarks;

    {
        Benchmark b;
        b.name = "DE_CMAES_Sphere";
        b.problem = std::make_unique<wrappers::problems::SphereProblem>(10);
        b.ea = std::make_unique<pagmo_wrappers::PagmoDifferentialEvolutionFactory>();
        b.hoa = std::make_unique<pagmo_wrappers::PagmoCmaesHyperOptimizer>();
        b.hoa_params.emplace("generations", static_cast<std::int64_t>(full_mode ? 30 : 8));
        b.algo_budget.generations = full_mode ? 100 : 30;
        b.opt_budget.generations = full_mode ? 30 : 8;
        b.opt_budget.function_evaluations = full_mode ? 5000 : 1200;
        b.trials = full_mode ? 3 : 1;
        benchmarks.push_back(std::move(b));
    }

    {
        Benchmark b;
        b.name = "PSO_PSOHyper_Rastrigin";
        b.problem = std::make_unique<wrappers::problems::RastriginProblem>(8);
        b.ea = std::make_unique<pagmo_wrappers::PagmoParticleSwarmOptimizationFactory>();
        b.hoa = std::make_unique<pagmo_wrappers::PagmoPsoHyperOptimizer>();
        b.hoa_params.emplace("generations", static_cast<std::int64_t>(full_mode ? 25 : 6));
        b.algo_budget.generations = full_mode ? 80 : 24;
        b.opt_budget.generations = full_mode ? 25 : 6;
        b.opt_budget.function_evaluations = full_mode ? 4000 : 1000;
        b.trials = full_mode ? 3 : 1;
        benchmarks.push_back(std::move(b));
    }

    {
        Benchmark b;
        b.name = "SADE_SA_Rosenbrock";
        b.problem = std::make_unique<wrappers::problems::RosenbrockProblem>(6);
        b.ea = std::make_unique<pagmo_wrappers::PagmoSelfAdaptiveDEFactory>();
        b.hoa = std::make_unique<pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer>();
        b.hoa_params.emplace("iterations", static_cast<std::int64_t>(full_mode ? 40 : 10));
        b.algo_budget.generations = full_mode ? 120 : 36;
        b.opt_budget.function_evaluations = full_mode ? 3000 : 900;
        b.trials = full_mode ? 3 : 1;
        benchmarks.push_back(std::move(b));
    }

    for (auto &b : benchmarks) {
        std::cout << "benchmark: " << b.name << "\n";
        std::cout << "  problem: " << b.problem->metadata().id << " dim=" << b.problem->dimension() << "\n";
        std::cout << "  ea: " << b.ea->identity().family << "\n";
        std::cout << "  hoa: " << b.hoa->identity().family << "\n";
        std::cout << "  trials: " << b.trials << "\n";

        b.hoa->configure(b.hoa_params);

        core::ExperimentConfig cfg;
        cfg.experiment_id = b.name;
        cfg.trials_per_optimizer = b.trials;
        cfg.islands = 1;
        cfg.algorithm_budget = b.algo_budget;
        cfg.optimizer_budget = b.opt_budget;
        cfg.log_file_path = b.name + "_benchmark.jsonl";

        if (std::filesystem::exists(cfg.log_file_path)) {
            std::filesystem::remove(cfg.log_file_path);
        }

        core::JsonlLogger logger(cfg.log_file_path);
        core::SequentialExperimentManager manager;

        auto t0 = std::chrono::steady_clock::now();
        auto result = manager.run_experiment(cfg, *b.hoa, *b.ea, *b.problem, logger);
        auto t1 = std::chrono::steady_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::vector<double> objectives;
        std::size_t total_trials = 0;
        std::size_t total_fevals = 0;

        for (const auto &r : result.optimizer_results) {
            objectives.push_back(r.best_objective);
            total_trials += r.trials.size();
            total_fevals += r.optimizer_usage.objective_calls;
        }

        auto stats = compute_stats(objectives);

        std::cout << "  results:\n";
        std::cout << "    time: " << ms << " ms\n";
        std::cout << "    trials: " << total_trials << "\n";
        std::cout << "    fevals: " << total_fevals << "\n";
        std::cout << "    objective (min/avg/max): " << stats.min << " / " << stats.avg << " / " << stats.max << "\n\n";
    }

    std::cout << "benchmark suite complete\n";
    return 0;
}

