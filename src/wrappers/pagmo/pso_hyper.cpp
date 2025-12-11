#include "hpoea/wrappers/pagmo/pso_hyper.hpp"

#include "hyper_tuning_udp.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/pso.hpp>
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

constexpr std::string_view kFamily = "PSOHyperOptimizer";
constexpr std::string_view kImplementation = "pagmo::pso";
constexpr std::string_view kVersion = "2.x";

ParameterSpace make_parameter_space() {
  ParameterSpace space;

  ParameterDescriptor variant;
  variant.name = "variant";
  variant.type = ParameterType::Integer;
  variant.integer_range = hpoea::core::IntegerRange{1, 6};
  variant.default_value = static_cast<std::int64_t>(5);
  space.add_descriptor(variant);

  ParameterDescriptor generations;
  generations.name = "generations";
  generations.type = ParameterType::Integer;
  generations.integer_range = hpoea::core::IntegerRange{1, 1000};
  generations.default_value = static_cast<std::int64_t>(100);
  space.add_descriptor(generations);

  ParameterDescriptor omega;
  omega.name = "omega";
  omega.type = ParameterType::Continuous;
  omega.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
  omega.default_value = 0.7298;
  space.add_descriptor(omega);

  ParameterDescriptor eta1;
  eta1.name = "eta1";
  eta1.type = ParameterType::Continuous;
  eta1.continuous_range = hpoea::core::ContinuousRange{0.0, 5.0};
  eta1.default_value = 2.05;
  space.add_descriptor(eta1);

  ParameterDescriptor eta2;
  eta2.name = "eta2";
  eta2.type = ParameterType::Continuous;
  eta2.continuous_range = hpoea::core::ContinuousRange{0.0, 5.0};
  eta2.default_value = 2.05;
  space.add_descriptor(eta2);

  ParameterDescriptor max_velocity;
  max_velocity.name = "max_velocity";
  max_velocity.type = ParameterType::Continuous;
  max_velocity.continuous_range = hpoea::core::ContinuousRange{0.0, 100.0};
  max_velocity.default_value = 0.5;
  space.add_descriptor(max_velocity);

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

PagmoPsoHyperOptimizer::PagmoPsoHyperOptimizer()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoPsoHyperOptimizer::configure(const ParameterSet &parameters) {
  configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

core::HyperparameterOptimizationResult PagmoPsoHyperOptimizer::optimize(
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

    HyperTuningUdp udp{context};

    const auto [lower, upper] = udp.get_bounds();
    pagmo::problem tuning_problem{udp};

    auto generations = static_cast<unsigned>(
        std::get<std::int64_t>(configured_parameters_.at("generations")));
    if (budget.generations.has_value()) {
      generations = std::min<unsigned>(generations, budget.generations.value());
    }

    const auto omega = std::get<double>(configured_parameters_.at("omega"));
    const auto eta1 = std::get<double>(configured_parameters_.at("eta1"));
    const auto eta2 = std::get<double>(configured_parameters_.at("eta2"));
    const auto max_velocity =
        std::get<double>(configured_parameters_.at("max_velocity"));
    const auto variant = static_cast<unsigned>(
        std::get<std::int64_t>(configured_parameters_.at("variant")));

    const auto seed32 =
        static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());

    // pso constructor parameters: gen, omega, eta1, eta2, max_vel, variant,
    // neighb_type, neighb_param, memory, seed
    pagmo::algorithm algorithm{pagmo::pso(generations, omega, eta1, eta2,
                                          max_velocity, variant, 2u, 4u, false,
                                          seed32)};

    const auto dimension = lower.size();
    const auto population_size = static_cast<pagmo::population::size_type>(
        std::max<std::size_t>(dimension * 4, dimension + 1));

    pagmo::population population{tuning_problem, population_size, seed32};

    const auto start_time = std::chrono::steady_clock::now();
    population = algorithm.evolve(population);
    const auto end_time = std::chrono::steady_clock::now();

    result.status = RunStatus::Success;
    result.trials = context->trials ? std::move(*context->trials)
                                    : std::vector<HyperparameterTrialRecord>{};

    if (context->best_trial.has_value()) {
      result.best_parameters = context->best_trial->parameters;
      result.best_objective =
          context->best_trial->optimization_result.best_fitness;
    } else {
      result.best_objective = population.champion_f()[0];
    }

    result.budget_usage.wall_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                              start_time);
    result.budget_usage.generations = generations;
    result.budget_usage.function_evaluations = context->evaluations;
    result.effective_optimizer_parameters = configured_parameters_;
    result.message = "Hyperparameter optimization completed.";
  } catch (const std::exception &ex) {
    result.status = RunStatus::InternalError;
    result.message = ex.what();
  }

  return result;
}

} // namespace hpoea::pagmo_wrappers
