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
#include <cmath>
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

// fixed seed keeps benchmark runs reproducible
constexpr unsigned long benchmark_seed = 1729;
// function_evaluations is the only budget we can compare across optimizers
constexpr std::size_t optimizer_fevals = 240;

struct Stats {
    double min, avg, max;
};

// empty input (every optimizer excluded) reports zeros
Stats compute_stats(const std::vector<double> &v) {
    if (v.empty()) {
        return {0.0, 0.0, 0.0};
    }
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
        b.opt_budget.function_evaluations = optimizer_fevals;
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
        b.opt_budget.function_evaluations = optimizer_fevals;
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
        // sa cost/evolve = n_T_adj*n_range_adj*bin_size*dim = 5*1*3*7 = 105
        // so 2 evolves fit 240
        b.hoa_params.emplace("n_T_adj", static_cast<std::int64_t>(5));
        b.hoa_params.emplace("n_range_adj", static_cast<std::int64_t>(1));
        b.hoa_params.emplace("bin_size", static_cast<std::int64_t>(3));
        b.algo_budget.generations = full_mode ? 120 : 36;
        b.opt_budget.function_evaluations = optimizer_fevals;
        b.trials = full_mode ? 3 : 1;
        benchmarks.push_back(std::move(b));
    }

    for (auto &b : benchmarks) {
        std::cout << "benchmark: " << b.name << "\n";
        std::cout << "  problem: " << b.problem->metadata().id << " dim=" << b.problem->dimension() << "\n";
        std::cout << "  ea: " << b.ea->identity().family << "\n";
        std::cout << "  hoa: " << b.hoa->identity().family << "\n";
        std::cout << "  trials: " << b.trials << "\n";

        core::ExperimentConfig cfg;
        cfg.experiment_id = b.name;
        cfg.trials_per_optimizer = b.trials;
        cfg.max_parallel_trials = 1;
        cfg.algorithm_budget = b.algo_budget;
        cfg.optimizer_budget = b.opt_budget;
        cfg.log_file_path = b.name + "_benchmark.jsonl";
        cfg.random_seed = benchmark_seed;
        cfg.optimizer_parameters = b.hoa_params;

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
        std::size_t excluded_trials = 0;

        for (const auto &r : result.optimizer_results) {
            const bool success_like = r.status == core::RunStatus::Success
                || r.status == core::RunStatus::BudgetExceeded;
            const bool finite = std::isfinite(r.best_objective);
            if (success_like && finite) {
                objectives.push_back(r.best_objective);
            } else {
                excluded_trials += 1;
            }
            total_trials += r.trials.size();
            total_fevals += r.optimizer_usage.objective_calls;
        }

        auto stats = compute_stats(objectives);

        std::cout << "  results:\n";
        std::cout << "    seed: " << result.actual_seed << "\n";
        std::cout << "    time: " << ms << " ms\n";
        std::cout << "    trials: " << total_trials << "\n";
        std::cout << "    fevals: " << total_fevals << "\n";
        std::cout << "    objective (min/avg/max): " << stats.min
                  << " / " << stats.avg << " / " << stats.max << "\n";
        std::cout << "    included_optimizers: " << objectives.size() << "\n";
        std::cout << "    excluded_optimizers: " << excluded_trials << "\n\n";
    }

    std::cout << "benchmark suite complete\n";
    return 0;
}

