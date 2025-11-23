#include "hpoea/wrappers/pagmo/de1220_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"
#include "hpoea/core/types.hpp"

#include <iostream>
#include <iomanip>

int main() {
    using namespace hpoea;

    wrappers::problems::AckleyProblem problem(10);
    pagmo_wrappers::PagmoDe1220Factory factory;
    auto algorithm = factory.create();

    core::ParameterSet params;
    params.emplace("population_size", static_cast<std::int64_t>(50));
    params.emplace("generations", static_cast<std::int64_t>(250));
    params.emplace("ftol", 1e-6);
    params.emplace("xtol", 1e-6);
    params.emplace("variant_adaptation", static_cast<std::int64_t>(1));
    params.emplace("memory", false);
    algorithm->configure(params);

    core::Budget budget;
    budget.generations = 250;

    auto result = algorithm->run(problem, budget, 999UL);

    if (result.status == core::RunStatus::Success) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "best_fitness: " << result.best_fitness << "\n";
        std::cout << "function_evaluations: " << result.budget_usage.function_evaluations << "\n";
        std::cout << "generations: " << result.budget_usage.generations << "\n";
        std::cout << "wall_time_ms: " << result.budget_usage.wall_time.count() << "\n";
    } else {
        std::cerr << "error: " << result.message << "\n";
        return 1;
    }

    return 0;
}


