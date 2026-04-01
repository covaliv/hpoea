#include "hpoea/core/baseline_optimizer.hpp"
#include "hpoea/core/error_classification.hpp"

#include <chrono>
#include <stdexcept>

namespace hpoea::core {

HyperparameterOptimizationResult BaselineOptimizer::optimize(
    const IEvolutionaryAlgorithmFactory &algorithm_factory,
    const IProblem &problem,
    const Budget & /*optimizer_budget*/,
    const Budget &algorithm_budget,
    unsigned long seed) {

    HyperparameterOptimizationResult result;
    result.seed = seed;

    const auto start_time = std::chrono::steady_clock::now();

    try {
        auto algorithm = algorithm_factory.create();

        auto params = algorithm_factory.parameter_space().apply_defaults(
            fixed_parameters_.value_or(ParameterSet{}));

        algorithm->configure(params);
        auto run_result = algorithm->run(problem, algorithm_budget, seed);

        const auto end_time = std::chrono::steady_clock::now();

        HyperparameterTrialRecord trial;
        trial.parameters = params;
        trial.optimization_result = std::move(run_result);
        result.trials.push_back(std::move(trial));

        const auto &best = result.trials.front();
        result.best_parameters = best.parameters;
        result.best_objective = best.optimization_result.best_fitness;
        result.status = best.optimization_result.status;
        result.optimizer_usage.objective_calls = 1;
        result.optimizer_usage.iterations = 0;
        result.optimizer_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.message = fixed_parameters_.has_value()
            ? "baseline run with fixed parameters"
            : "baseline run with default parameters";
    } catch (const std::exception &ex) {
        const auto end_time = std::chrono::steady_clock::now();
        result.optimizer_usage.wall_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        result.message = ex.what();

        const auto classified = classify_exception(ex);
        result.status = classified.status;
        result.error_info = classified.error_info;
    }

    return result;
}

} // namespace hpoea::core
