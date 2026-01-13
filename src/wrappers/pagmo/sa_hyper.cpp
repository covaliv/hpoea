#include "hpoea/wrappers/pagmo/sa_hyper.hpp"

#include "hyper_tuning_udp.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/simulated_annealing.hpp>
#include <pagmo/population.hpp>
#include <pagmo/problem.hpp>

namespace {

using hpoea::core::AlgorithmIdentity;
using hpoea::core::Budget;
using hpoea::core::HyperparameterOptimizationResult;
using hpoea::core::HyperparameterTrialRecord;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSet;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;
using hpoea::core::RunStatus;
using hpoea::pagmo_wrappers::HyperTuningUdp;

constexpr std::string_view kFamily = "SimulatedAnnealing";
constexpr std::string_view kImplementation = "pagmo::simulated_annealing";
constexpr std::string_view kVersion = "2.x";

ParameterSpace make_parameter_space() {
  ParameterSpace space;

  ParameterDescriptor iterations;
  iterations.name = "iterations";
  iterations.type = ParameterType::Integer;
  iterations.integer_range = hpoea::core::IntegerRange{1, 100000};
  iterations.default_value = static_cast<std::int64_t>(1000);
  space.add_descriptor(iterations);

  ParameterDescriptor ts;
  ts.name = "ts";
  ts.type = ParameterType::Continuous;
  ts.continuous_range = hpoea::core::ContinuousRange{1e-6, 100.0};
  ts.default_value = 10.0;
  space.add_descriptor(ts);

  ParameterDescriptor tf;
  tf.name = "tf";
  tf.type = ParameterType::Continuous;
  tf.continuous_range = hpoea::core::ContinuousRange{1e-6, 100.0};
  tf.default_value = 0.1;
  space.add_descriptor(tf);

  ParameterDescriptor n_T_adj;
  n_T_adj.name = "n_T_adj";
  n_T_adj.type = ParameterType::Integer;
  n_T_adj.integer_range = hpoea::core::IntegerRange{1, 10000};
  n_T_adj.default_value = static_cast<std::int64_t>(10);
  space.add_descriptor(n_T_adj);

  ParameterDescriptor n_range_adj;
  n_range_adj.name = "n_range_adj";
  n_range_adj.type = ParameterType::Integer;
  n_range_adj.integer_range = hpoea::core::IntegerRange{1, 10000};
  n_range_adj.default_value = static_cast<std::int64_t>(1);
  space.add_descriptor(n_range_adj);

  ParameterDescriptor bin_size;
  bin_size.name = "bin_size";
  bin_size.type = ParameterType::Integer;
  bin_size.integer_range = hpoea::core::IntegerRange{1, 1000};
  bin_size.default_value = static_cast<std::int64_t>(10);
  space.add_descriptor(bin_size);

  ParameterDescriptor start_range;
  start_range.name = "start_range";
  start_range.type = ParameterType::Continuous;
  start_range.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
  start_range.default_value = 1.0;
  space.add_descriptor(start_range);

  return space;
}

AlgorithmIdentity make_identity() {
  AlgorithmIdentity id;
  id.family = std::string{kFamily};
  id.implementation = std::string{kImplementation};
  id.version = std::string{kVersion};
  return id;
}

} // namespace

namespace hpoea::pagmo_wrappers {

PagmoSimulatedAnnealingHyperOptimizer::PagmoSimulatedAnnealingHyperOptimizer()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoSimulatedAnnealingHyperOptimizer::configure(
    const ParameterSet &parameters) {
  configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

void PagmoSimulatedAnnealingHyperOptimizer::set_search_space(
    std::shared_ptr<core::SearchSpace> search_space) {
  search_space_ = std::move(search_space);
}

core::HyperparameterOptimizationResult
PagmoSimulatedAnnealingHyperOptimizer::optimize(
    const core::IEvolutionaryAlgorithmFactory &algorithm_factory,
    const core::IProblem &problem, const Budget &budget, unsigned long seed) {

  HyperparameterOptimizationResult result;
  result.status = RunStatus::InternalError;
  result.seed = seed;

  try {
    auto context = std::make_shared<HyperTuningUdp::Context>();
    context->factory = &algorithm_factory;
    context->problem = &problem;
    context->algorithm_budget = budget;
    context->base_seed = seed;
    context->trials =
        std::make_shared<std::vector<HyperparameterTrialRecord>>();
    context->search_space = search_space_;

    HyperTuningUdp udp{context};

    const auto [lower, upper] = udp.get_bounds();
    pagmo::problem tuning_problem{udp};

    const auto ts = std::get<double>(configured_parameters_.at("ts"));
    const auto tf = std::get<double>(configured_parameters_.at("tf"));
    const auto n_T_adj = static_cast<unsigned>(
        std::get<std::int64_t>(configured_parameters_.at("n_T_adj")));
    const auto n_range_adj = static_cast<unsigned>(
        std::get<std::int64_t>(configured_parameters_.at("n_range_adj")));
    const auto bin_size = static_cast<unsigned>(
        std::get<std::int64_t>(configured_parameters_.at("bin_size")));
    const auto start_range =
        std::get<double>(configured_parameters_.at("start_range"));

    const auto seed32 =
        static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());

    pagmo::simulated_annealing sa_alg(ts, tf, n_T_adj, n_range_adj, bin_size,
                                      start_range, seed32);
    pagmo::algorithm algorithm{sa_alg};

    pagmo::population population{tuning_problem, 1, seed32};

    auto iterations = static_cast<unsigned>(
        std::get<std::int64_t>(configured_parameters_.at("iterations")));
    const auto dimension = lower.size();
    const auto evals_per_evolve = n_T_adj * n_range_adj * bin_size * dimension;

    if (budget.function_evaluations.has_value()) {
      const auto max_evolves =
          static_cast<unsigned>(budget.function_evaluations.value() /
                                std::max<std::size_t>(evals_per_evolve, 1));
      iterations = std::min(iterations, max_evolves);
    }

    const auto start_time = std::chrono::steady_clock::now();
    // call evolve() multiple times to reach desired iteration count
    for (unsigned i = 0; i < iterations; ++i) {
      population = algorithm.evolve(population);
      // check budget after each evolve
      if (budget.function_evaluations.has_value() &&
          context->get_evaluations() >= budget.function_evaluations.value()) {
        break;
      }
    }
    const auto end_time = std::chrono::steady_clock::now();

    result.status = RunStatus::Success;
    result.trials = context->trials ? std::move(*context->trials)
                                    : std::vector<HyperparameterTrialRecord>{};

    const auto best_trial = context->get_best_trial();
    if (best_trial.has_value()) {
      result.best_parameters = best_trial->parameters;
      result.best_objective = best_trial->optimization_result.best_fitness;
    } else {
      result.best_objective = population.champion_f()[0];
    }

    result.budget_usage.wall_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    result.budget_usage.generations = iterations; // number of evolve() calls
    result.budget_usage.function_evaluations = context->get_evaluations();
    result.effective_optimizer_parameters = configured_parameters_;
    result.message = "Hyperparameter optimization completed.";
  } catch (const std::exception &ex) {
    result.status = RunStatus::InternalError;
    result.message = ex.what();
  }

  return result;
}

} // namespace hpoea::pagmo_wrappers
