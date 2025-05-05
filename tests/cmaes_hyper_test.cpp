#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <variant>
#include <vector>

namespace {

class SphereProblem final : public hpoea::core::IProblem {
public:
    explicit SphereProblem(std::size_t dimension) {
        metadata_.id = "sphere";
        metadata_.family = "benchmark";
        metadata_.description = "Simple sphere function";
        dimension_ = dimension;
        lower_bounds_.assign(dimension_, -5.0);
        upper_bounds_.assign(dimension_, 5.0);
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override {
        double sum = 0.0;
        for (const auto value : decision_vector) {
            sum += value * value;
        }
        return sum;
    }

private:
    hpoea::core::ProblemMetadata metadata_{};
    std::size_t dimension_{0};
    std::vector<double> lower_bounds_{};
    std::vector<double> upper_bounds_{};
};

} // namespace

int main() {
    using namespace hpoea;

    const char *run_flag = std::getenv("HPOEA_RUN_CMAES_TESTS");
    if (!run_flag || std::string_view{run_flag} != "1") {
        std::cout << "Skipping CMA-ES hyper optimizer test (set HPOEA_RUN_CMAES_TESTS=1 to enable)" << std::endl;
        return 0;
    }

    const bool verbose = [] {
        if (const char *flag = std::getenv("HPOEA_LOG_RESULTS")) {
            return std::string_view{flag} == "1";
        }
        return false;
    }();

    SphereProblem problem{5};
    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;

    core::ParameterSet optimizer_overrides;
    optimizer_overrides.emplace("generations", static_cast<std::int64_t>(40));
    optimizer_overrides.emplace("sigma0", 0.8);
    optimizer_overrides.emplace("ftol", 1e-4);
    optimizer.configure(optimizer_overrides);

    core::Budget budget;
    budget.generations = 40;
    budget.function_evaluations = 40000;

    const unsigned long seed = 1337UL;
    const auto result = optimizer.optimize(factory, problem, budget, seed);

    if (result.status != core::RunStatus::Success) {
        std::cerr << "CMA-ES optimization failed: " << result.message << '\n';
        return 1;
    }

    if (result.trials.empty()) {
        std::cerr << "Expected trials to be populated" << '\n';
        return 2;
    }

    if (result.best_objective > 5.0) {
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

    const auto expected_effective = optimizer.parameter_space().apply_defaults(optimizer_overrides);
    if (result.effective_optimizer_parameters != expected_effective) {
        std::cerr << "Effective optimizer parameters differ from expected defaults" << '\n';
        return 6;
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
            return 7;
        }
    }

    bool has_population_param = false;
    if (const auto iter = result.best_parameters.find("population_size"); iter != result.best_parameters.end()) {
        has_population_param = true;
        if (!std::holds_alternative<std::int64_t>(iter->second)) {
            std::cerr << "population_size parameter not stored as integer" << '\n';
            return 8;
        }
    }

    if (!has_population_param) {
        std::cerr << "Best parameters missing population_size" << '\n';
        return 9;
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

