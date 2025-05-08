#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <variant>
#include <vector>

int main() {
    using namespace hpoea;

    const char *run_flag = std::getenv("HPOEA_RUN_PSO_HYPER_TESTS");
    if (!run_flag || std::string_view{run_flag} != "1") {
        std::cout << "Skipping PSO hyper optimizer test (set HPOEA_RUN_PSO_HYPER_TESTS=1 to enable)" << std::endl;
        return 0;
    }

    const bool verbose = [] {
        if (const char *flag = std::getenv("HPOEA_LOG_RESULTS")) {
            return std::string_view{flag} == "1";
        }
        return false;
    }();

    wrappers::problems::SphereProblem problem{5};
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    pagmo_wrappers::PagmoPsoHyperOptimizer optimizer;

    core::ParameterSet optimizer_overrides;
    optimizer_overrides.emplace("generations", static_cast<std::int64_t>(30));
    optimizer_overrides.emplace("omega", 0.7298);
    optimizer_overrides.emplace("eta1", 2.05);
    optimizer_overrides.emplace("eta2", 2.05);
    optimizer.configure(optimizer_overrides);

    core::Budget budget;
    budget.generations = 30;
    budget.function_evaluations = 5000;

    const unsigned long seed = 1337UL;
    const auto result = optimizer.optimize(factory, problem, budget, seed);

    if (result.status != core::RunStatus::Success) {
        std::cerr << "PSO hyper optimization failed: " << result.message << '\n';
        return 1;
    }

    if (result.trials.empty()) {
        std::cerr << "Expected trials to be populated" << '\n';
        return 2;
    }

    if (result.best_objective > 10.0) {
        std::cerr << "Best hyperparameter objective too large: " << result.best_objective << '\n';
        return 3;
    }

    if (result.budget_usage.generations == 0
        || (budget.generations.has_value() && result.budget_usage.generations > budget.generations.value())) {
        std::cerr << "Unexpected generation usage: " << result.budget_usage.generations << '\n';
        return 4;
    }

    if (result.budget_usage.function_evaluations == 0
        || (budget.function_evaluations.has_value()
            && result.budget_usage.function_evaluations > budget.function_evaluations.value())) {
        std::cerr << "Unexpected function evaluation usage: " << result.budget_usage.function_evaluations << '\n';
        return 5;
    }

    for (const auto &trial : result.trials) {
        const auto status = trial.optimization_result.status;
        if (verbose) {
            std::cout << "trial.best_fitness=" << trial.optimization_result.best_fitness
                      << ", status=" << static_cast<int>(status)
                      << ", message='" << trial.optimization_result.message << "'" << '\n';
        }
        if (status != core::RunStatus::Success && status != core::RunStatus::BudgetExceeded) {
            std::cerr << "Encountered failed hyperparameter trial" << '\n';
            return 6;
        }
    }

    if (verbose) {
        std::cout << std::fixed << std::setprecision(6)
                  << "best_objective=" << result.best_objective
                  << ", trials=" << result.trials.size()
                  << ", generations_used=" << result.budget_usage.generations
                  << ", fevals_used=" << result.budget_usage.function_evaluations << '\n';

        for (const auto &[name, value] : result.best_parameters) {
            std::visit([&](const auto &typed_value) {
                std::cout << "  best_param." << name << " = " << typed_value << '\n';
            }, value);
        }
    }

    return 0;
}

