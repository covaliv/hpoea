#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <filesystem>
#include <iostream>
#include <string_view>

int main() {
    using namespace hpoea;

    const char *run_flag = std::getenv("HPOEA_RUN_PARALLEL_TESTS");
    if (!run_flag || std::string_view{run_flag} != "1") {
        std::cout << "Skipping parallel experiment manager test (set HPOEA_RUN_PARALLEL_TESTS=1 to enable)" << std::endl;
        return 0;
    }

    const bool verbose = [] {
        if (const char *flag = std::getenv("HPOEA_LOG_RESULTS")) {
            return std::string_view{flag} == "1";
        }
        return false;
    }();

    // Create test problem
    wrappers::problems::SphereProblem problem(5);

    // Create EA factory
    pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;

    // Create HOA
    pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;

    // Configure optimizer
    core::ParameterSet optimizer_params;
    optimizer_params.emplace("generations", static_cast<std::int64_t>(20));
    optimizer_params.emplace("sigma0", 0.5);
    optimizer.configure(optimizer_params);

    // Configure experiment
    core::ExperimentConfig config;
    config.experiment_id = "parallel_test";
    config.trials_per_optimizer = 4;
    config.islands = 2;
    config.algorithm_budget.generations = 50;
    config.optimizer_budget.generations = 20;
    config.optimizer_budget.function_evaluations = 2000;
    config.log_file_path = "parallel_test.jsonl";

    // Clean up old log file
    if (std::filesystem::exists(config.log_file_path)) {
        std::filesystem::remove(config.log_file_path);
    }

    // Create logger
    core::JsonlLogger logger(config.log_file_path);

    // Test sequential manager
    {
        core::SequentialExperimentManager sequential_manager;
        auto sequential_result = sequential_manager.run_experiment(config, optimizer, ea_factory, problem, logger);

        if (sequential_result.experiment_id != config.experiment_id) {
            std::cerr << "Sequential manager: experiment ID mismatch" << '\n';
            return 1;
        }

        if (sequential_result.optimizer_results.size() != config.trials_per_optimizer) {
            std::cerr << "Sequential manager: unexpected number of results" << '\n';
            return 2;
        }

        if (verbose) {
            std::cout << "Sequential manager: " << sequential_result.optimizer_results.size() << " trials completed" << '\n';
        }
    }

    // Test parallel manager
    {
        core::ParallelExperimentManager parallel_manager(2);
        auto parallel_result = parallel_manager.run_experiment(config, optimizer, ea_factory, problem, logger);

        if (parallel_result.experiment_id != config.experiment_id) {
            std::cerr << "Parallel manager: experiment ID mismatch" << '\n';
            return 3;
        }

        if (parallel_result.optimizer_results.size() != config.trials_per_optimizer) {
            std::cerr << "Parallel manager: unexpected number of results" << '\n';
            return 4;
        }

        if (verbose) {
            std::cout << "Parallel manager: " << parallel_result.optimizer_results.size() << " trials completed" << '\n';
        }

        // Verify all results have valid status
        for (const auto &opt_result : parallel_result.optimizer_results) {
            if (opt_result.status != core::RunStatus::Success && opt_result.status != core::RunStatus::BudgetExceeded) {
                std::cerr << "Parallel manager: found failed optimization result" << '\n';
                return 5;
            }
        }
    }

    // Verify log file exists and is readable
    if (!std::filesystem::exists(config.log_file_path)) {
        std::cerr << "Log file was not created" << '\n';
        return 6;
    }

    if (verbose) {
        std::cout << "Log file created: " << config.log_file_path << '\n';
    }

    // Clean up
    std::filesystem::remove(config.log_file_path);

    return 0;
}

