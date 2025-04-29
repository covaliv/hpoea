#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
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

class RosenbrockProblem final : public hpoea::core::IProblem {
public:
    explicit RosenbrockProblem(std::size_t dimension) {
        metadata_.id = "rosenbrock";
        metadata_.family = "benchmark";
        metadata_.description = "Rosenbrock valley";
        dimension_ = dimension;
        lower_bounds_.assign(dimension_, -5.0);
        upper_bounds_.assign(dimension_, 10.0);
    }

    [[nodiscard]] const hpoea::core::ProblemMetadata &metadata() const noexcept override { return metadata_; }

    [[nodiscard]] std::size_t dimension() const override { return dimension_; }

    [[nodiscard]] std::vector<double> lower_bounds() const override { return lower_bounds_; }

    [[nodiscard]] std::vector<double> upper_bounds() const override { return upper_bounds_; }

    [[nodiscard]] double evaluate(const std::vector<double> &decision_vector) const override {
        double sum = 0.0;
        for (std::size_t i = 0; i + 1 < decision_vector.size(); ++i) {
            const double xi = decision_vector[i];
            const double xnext = decision_vector[i + 1];
            const double term1 = 100.0 * std::pow(xnext - xi * xi, 2);
            const double term2 = std::pow(1.0 - xi, 2);
            sum += term1 + term2;
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

    const bool verbose = [] {
        if (const char *flag = std::getenv("HPOEA_LOG_RESULTS")) {
            return std::string_view{flag} == "1";
        }
        return false;
    }();

    pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;
    const std::vector<unsigned long> seeds{42UL, 1337UL, 2024UL, 9001UL, 123456UL};

    struct TestCase {
        std::string name;
        std::function<std::unique_ptr<hpoea::core::IProblem>()> make_problem;
        core::ParameterSet parameters;
        core::Budget budget;
        double max_allowed_fitness{0.0};
    };

    const auto run_test_case = [&](const TestCase &test_case) -> bool {
        double worst_fitness = 0.0;

        for (const auto seed : seeds) {
            auto problem = test_case.make_problem();
            if (!problem) {
                std::cerr << "Unable to construct problem for test case '" << test_case.name << "'" << '\n';
                return false;
            }

            auto algorithm = factory.create();
            algorithm->configure(test_case.parameters);

            const auto result = algorithm->run(*problem, test_case.budget, seed);

            if (result.status != core::RunStatus::Success) {
                std::cerr << "Test case '" << test_case.name << "' failed for seed " << seed
                          << " with error: " << result.message << '\n';
                return false;
            }

            if (result.best_solution.size() != problem->dimension()) {
                std::cerr << "Test case '" << test_case.name << "' returned invalid solution size for seed " << seed
                          << '\n';
                return false;
            }

            if (result.budget_usage.generations == 0
                || (test_case.budget.generations.has_value()
                    && result.budget_usage.generations > test_case.budget.generations.value())) {
                std::cerr << "Test case '" << test_case.name << "' used unexpected generation count for seed " << seed
                          << '\n';
                return false;
            }

            worst_fitness = std::max(worst_fitness, result.best_fitness);

            if (verbose) {
                std::cout << std::fixed << std::setprecision(6)
                          << "test=" << test_case.name << ", seed=" << seed
                          << ", best_fitness=" << result.best_fitness
                          << ", generations=" << result.budget_usage.generations
                          << ", fevals=" << result.budget_usage.function_evaluations << '\n';
            }
        }

        if (worst_fitness > test_case.max_allowed_fitness) {
            std::cerr << "Test case '" << test_case.name << "' worst fitness too large: " << worst_fitness << '\n';
            return false;
        }

        if (verbose) {
            std::cout << "test=" << test_case.name << ", worst_fitness=" << worst_fitness << '\n';
        }

        return true;
    };

    std::vector<TestCase> test_cases;

    {
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(120));
        params.emplace("generations", static_cast<std::int64_t>(350));
        params.emplace("scaling_factor", 0.7);
        params.emplace("crossover_rate", 0.9);

        core::Budget budget;
        budget.generations = 400;
        budget.function_evaluations = 50000;

        test_cases.push_back(TestCase{
            "sphere",
            [] { return std::make_unique<SphereProblem>(10); },
            std::move(params),
            budget,
            5e-3});
    }

    {
        core::ParameterSet params;
        params.emplace("population_size", static_cast<std::int64_t>(150));
        params.emplace("generations", static_cast<std::int64_t>(500));
        params.emplace("scaling_factor", 0.6);
        params.emplace("crossover_rate", 0.85);

        core::Budget budget;
        budget.generations = 600;
        budget.function_evaluations = 80000;

        test_cases.push_back(TestCase{
            "rosenbrock",
            [] { return std::make_unique<RosenbrockProblem>(6); },
            std::move(params),
            budget,
            1.0});
    }

    for (const auto &test_case : test_cases) {
        if (!run_test_case(test_case)) {
            return 1;
        }
    }

    auto algorithm = factory.create();
    try {
        core::ParameterSet invalid;
        invalid.emplace("variant", static_cast<std::int64_t>(0));
        algorithm->configure(invalid);
        std::cerr << "Expected ParameterValidationError for invalid variant" << '\n';
        return 5;
    } catch (const core::ParameterValidationError &) {
        // Expected
    }

    return 0;
}

