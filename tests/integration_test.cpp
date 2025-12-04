#include "hpoea/core/experiment.hpp"
#include "hpoea/core/logging.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/pso_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/nm_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

int main() {
    using namespace hpoea;

    const bool verbose = [] {
        if (const char *flag = std::getenv("HPOEA_LOG_RESULTS")) {
            return std::string_view{flag} == "1";
        }
        return false;
    }();

    std::vector<std::pair<std::string, std::unique_ptr<core::IProblem>>> problems;
    problems.emplace_back("sphere", std::make_unique<wrappers::problems::SphereProblem>(5));
    problems.emplace_back("rosenbrock", std::make_unique<wrappers::problems::RosenbrockProblem>(5));
    problems.emplace_back("rastrigin", std::make_unique<wrappers::problems::RastriginProblem>(5));

    std::vector<std::pair<std::string, std::unique_ptr<core::IEvolutionaryAlgorithmFactory>>> ea_factories;
    ea_factories.emplace_back("DE", std::make_unique<pagmo_wrappers::PagmoDifferentialEvolutionFactory>());
    ea_factories.emplace_back("PSO", std::make_unique<pagmo_wrappers::PagmoParticleSwarmOptimizationFactory>());
    ea_factories.emplace_back("SADE", std::make_unique<pagmo_wrappers::PagmoSelfAdaptiveDEFactory>());

    std::vector<std::pair<std::string, std::unique_ptr<core::IHyperparameterOptimizer>>> hoas;
    hoas.emplace_back("CMA-ES", std::make_unique<pagmo_wrappers::PagmoCmaesHyperOptimizer>());
    hoas.emplace_back("SA", std::make_unique<pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer>());
    hoas.emplace_back("PSO-Hyper", std::make_unique<pagmo_wrappers::PagmoPsoHyperOptimizer>());
    hoas.emplace_back("Nelder-Mead", std::make_unique<pagmo_wrappers::PagmoNelderMeadHyperOptimizer>());

    int failures = 0;
    int successes = 0;

    for (const auto &[problem_name, problem] : problems) {
        for (const auto &[ea_name, ea_factory] : ea_factories) {
            for (const auto &[hoa_name, hoa] : hoas) {
                const std::string experiment_id = problem_name + "_" + ea_name + "_" + hoa_name;

                if (verbose) {
                    std::cout << "Testing: " << experiment_id << '\n';
                }

                try {
                    // Configure HOA
                    core::ParameterSet hoa_params;
                    if (hoa_name == "CMA-ES") {
                        hoa_params.emplace("generations", static_cast<std::int64_t>(10));
                    } else if (hoa_name == "SA") {
                        hoa_params.emplace("iterations", static_cast<std::int64_t>(20));
                    } else if (hoa_name == "PSO-Hyper") {
                        hoa_params.emplace("generations", static_cast<std::int64_t>(10));
                        hoa_params.emplace("omega", 0.7298);
                        hoa_params.emplace("eta1", 2.05);
                        hoa_params.emplace("eta2", 2.05);
                        hoa_params.emplace("max_velocity", 0.5);
                        hoa_params.emplace("variant", static_cast<std::int64_t>(5));
                    } else if (hoa_name == "Nelder-Mead") {
                        hoa_params.emplace("max_fevals", static_cast<std::int64_t>(50));
                        hoa_params.emplace("xtol_rel", 1e-6);
                        hoa_params.emplace("ftol_rel", 1e-6);
                    }
                    hoa->configure(hoa_params);

                    // Configure experiment
                    core::ExperimentConfig config;
                    config.experiment_id = experiment_id;
                    config.trials_per_optimizer = 1;
                    config.islands = 1;
                    config.algorithm_budget.generations = 30;
                    config.optimizer_budget.generations = 10;
                    config.optimizer_budget.function_evaluations = 1000;
                    config.log_file_path = "integration_test.jsonl";

                    // Create logger
                    core::JsonlLogger logger(config.log_file_path);

                    // Run experiment
                    core::SequentialExperimentManager manager;
                    auto result = manager.run_experiment(config, *hoa, *ea_factory, *problem, logger);

                    if (result.experiment_id != experiment_id) {
                        std::cerr << "Experiment ID mismatch for " << experiment_id << '\n';
                        failures++;
                        continue;
                    }

                    if (result.optimizer_results.empty()) {
                        std::cerr << "No results for " << experiment_id << '\n';
                        failures++;
                        continue;
                    }

                    const auto &opt_result = result.optimizer_results[0];
                    if (opt_result.status != core::RunStatus::Success
                        && opt_result.status != core::RunStatus::BudgetExceeded) {
                        std::cerr << "Failed optimization for " << experiment_id << ": " << opt_result.message << '\n';
                        failures++;
                        continue;
                    }

                    successes++;

                    if (verbose) {
                        std::cout << "  Success: best_objective=" << opt_result.best_objective
                                  << ", trials=" << opt_result.trials.size() << '\n';
                    }

                } catch (const std::exception &ex) {
                    std::cerr << "Exception for " << experiment_id << ": " << ex.what() << '\n';
                    failures++;
                }
            }
        }
    }

    // Clean up
    if (std::filesystem::exists("integration_test.jsonl")) {
        std::filesystem::remove("integration_test.jsonl");
    }

    if (verbose) {
        std::cout << "\nIntegration test summary:\n";
        std::cout << "  Successes: " << successes << '\n';
        std::cout << "  Failures: " << failures << '\n';
    }

    return failures > 0 ? 1 : 0;
}

