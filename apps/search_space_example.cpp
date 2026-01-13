#include "hpoea/core/search_space.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <iomanip>
#include <iostream>
#include <memory>

int main() {
  using namespace hpoea;

  wrappers::problems::RosenbrockProblem problem(8);
  pagmo_wrappers::PagmoDifferentialEvolutionFactory ea_factory;

  // create search space to customize hyperparameter optimization
  auto search = std::make_shared<core::SearchSpace>();

  // fix population_size at 100 (won't be tuned)
  search->fix("population_size", std::int64_t{100});

  // optimize scaling_factor with narrower bounds [0.3, 0.9]
  search->optimize("scaling_factor", core::ContinuousRange{0.3, 0.9});

  // optimize crossover_rate with narrower bounds [0.7, 1.0]
  search->optimize("crossover_rate", core::ContinuousRange{0.7, 1.0});

  // only try specific variant values
  search->optimize_choices("variant",
                           {std::int64_t{1}, std::int64_t{2}, std::int64_t{5}});

  // exclude ftol from optimization (use algorithm default)
  search->exclude("ftol");

  // configure optimizer
  pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
  optimizer.set_search_space(search);

  core::ParameterSet optimizer_params;
  optimizer_params.emplace("generations", static_cast<std::int64_t>(15));
  optimizer_params.emplace("sigma0", 0.3);
  optimizer.configure(optimizer_params);

  core::Budget budget;
  budget.generations = 15;
  budget.function_evaluations = 5000;

  std::cout << "running hpo with custom search space...\n";
  auto result = optimizer.optimize(ea_factory, problem, budget, 42UL);

  if (result.status == core::RunStatus::Success) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nbest objective: " << result.best_objective << "\n";
    std::cout << "trials completed: " << result.trials.size() << "\n";

    std::cout << "\nbest parameters:\n";
    for (const auto &[name, value] : result.best_parameters) {
      std::cout << "  " << name << ": ";
      std::visit([](auto v) { std::cout << v; }, value);
      std::cout << "\n";
    }

    std::cout << "\nbudget usage:\n";
    std::cout << "  function_evaluations: "
              << result.budget_usage.function_evaluations << "\n";
    std::cout << "  wall_time_ms: " << result.budget_usage.wall_time.count()
              << "\n";
  } else {
    std::cerr << "error: " << result.message << "\n";
    return 1;
  }

  return 0;
}
