#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/nm_hyper.hpp"
#include "hpoea/wrappers/pagmo/pso_hyper.hpp"
#include "hpoea/wrappers/pagmo/sa_hyper.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <variant>

using namespace hpoea;

namespace {

const bool verbose = [] {
    if (const char *v = std::getenv("HPOEA_LOG_RESULTS")) {
        return std::string_view{v} == "1";
    }
    return false;
}();

// test a hyperparameter optimizer
template <typename HOA>
bool test_hoa(const std::string &name, HOA &hoa, core::ParameterSet &params,
              core::Budget &budget, double max_objective) {
    wrappers::problems::SphereProblem problem{5};
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;

    hoa.configure(params);
    auto result = hoa.optimize(factory, problem, budget, 1337UL);

    if (result.status != core::RunStatus::Success) {
        std::cerr << name << " failed: " << result.message << "\n";
        return false;
    }

    if (result.trials.empty()) {
        std::cerr << name << " no trials generated\n";
        return false;
    }

    if (result.best_objective > max_objective) {
        std::cerr << name << " objective=" << result.best_objective
                  << " exceeds limit=" << max_objective << "\n";
        return false;
    }

    // check budget not exceeded
    if (budget.function_evaluations &&
        result.budget_usage.function_evaluations > *budget.function_evaluations) {
        std::cerr << name << " exceeded feval budget\n";
        return false;
    }

    if (budget.generations && result.budget_usage.generations > *budget.generations) {
        std::cerr << name << " exceeded generation budget\n";
        return false;
    }

    // verify trials have valid status
    for (const auto &trial : result.trials) {
        auto status = trial.optimization_result.status;
        if (status != core::RunStatus::Success && status != core::RunStatus::BudgetExceeded) {
            std::cerr << name << " invalid trial status\n";
            return false;
        }
    }

    if (verbose) {
        std::cout << std::fixed << std::setprecision(6)
                  << name << " objective=" << result.best_objective
                  << " trials=" << result.trials.size() << "\n";

        for (const auto &[pname, value] : result.best_parameters) {
            std::visit([&](const auto &v) {
                std::cout << "  " << pname << "=" << v << "\n";
            }, value);
        }
    }

    return true;
}

} // namespace

int main() {
    int failures = 0;

    // cmaes hyper optimizer
    std::cout << "cmaes hoa\n";
    {
        pagmo_wrappers::PagmoCmaesHyperOptimizer hoa;
        core::ParameterSet params;
        params.emplace("generations", static_cast<std::int64_t>(20));
        params.emplace("sigma0", 0.5);

        core::Budget budget;
        budget.generations = 20;
        budget.function_evaluations = 10000;

        if (!test_hoa("  cmaes", hoa, params, budget, 5.0)) failures++;
    }

    // simulated annealing hyper optimizer
    std::cout << "sa hoa\n";
    {
        pagmo_wrappers::PagmoSimulatedAnnealingHyperOptimizer hoa;
        core::ParameterSet params;
        params.emplace("iterations", static_cast<std::int64_t>(30));
        params.emplace("ts", 10.0);
        params.emplace("tf", 0.1);

        core::Budget budget;
        budget.function_evaluations = 5000;

        if (!test_hoa("  sa", hoa, params, budget, 10.0)) failures++;
    }

    // pso hyper optimizer
    std::cout << "pso hoa\n";
    {
        pagmo_wrappers::PagmoPsoHyperOptimizer hoa;
        core::ParameterSet params;
        params.emplace("generations", static_cast<std::int64_t>(20));

        core::Budget budget;
        budget.generations = 20;
        budget.function_evaluations = 5000;

        if (!test_hoa("  pso", hoa, params, budget, 10.0)) failures++;
    }

    // nelder-mead hyper optimizer
    std::cout << "nm hoa\n";
    {
        pagmo_wrappers::PagmoNelderMeadHyperOptimizer hoa;
        core::ParameterSet params;
        params.emplace("max_fevals", static_cast<std::int64_t>(30));

        core::Budget budget;
        budget.function_evaluations = 5000;

        if (!test_hoa("  nelder-mead", hoa, params, budget, 10.0)) failures++;
    }

    if (failures > 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << "all hoa tests passed\n";
    return 0;
}
