#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"

#include "hyper_tuning_udp.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/cmaes.hpp>
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

constexpr std::string_view kFamily = "CMAES";
constexpr std::string_view kImplementation = "pagmo::cmaes";
constexpr std::string_view kVersion = "2.x";

ParameterSpace make_parameter_space() {
  ParameterSpace space;

  ParameterDescriptor generations;
  generations.name = "generations";
  generations.type = ParameterType::Integer;
  generations.integer_range = hpoea::core::IntegerRange{1, 1000};
  generations.default_value = static_cast<std::int64_t>(100);
  space.add_descriptor(generations);

  ParameterDescriptor sigma0;
  sigma0.name = "sigma0";
  sigma0.type = ParameterType::Continuous;
  sigma0.continuous_range = hpoea::core::ContinuousRange{1e-6, 10.0};
  sigma0.default_value = 0.5;
  space.add_descriptor(sigma0);

  ParameterDescriptor cc;
  cc.name = "cc";
  cc.type = ParameterType::Continuous;
  cc.continuous_range = hpoea::core::ContinuousRange{-1.0, 1.0};
  cc.default_value = -1.0;
  space.add_descriptor(cc);

  ParameterDescriptor cs;
  cs.name = "cs";
  cs.type = ParameterType::Continuous;
  cs.continuous_range = hpoea::core::ContinuousRange{-1.0, 1.0};
  cs.default_value = -1.0;
  space.add_descriptor(cs);

  ParameterDescriptor c1;
  c1.name = "c1";
  c1.type = ParameterType::Continuous;
  c1.continuous_range = hpoea::core::ContinuousRange{-1.0, 1.0};
  c1.default_value = -1.0;
  space.add_descriptor(c1);

  ParameterDescriptor cmu;
  cmu.name = "cmu";
  cmu.type = ParameterType::Continuous;
  cmu.continuous_range = hpoea::core::ContinuousRange{-1.0, 1.0};
  cmu.default_value = -1.0;
  space.add_descriptor(cmu);

  ParameterDescriptor ftol;
  ftol.name = "ftol";
  ftol.type = ParameterType::Continuous;
  ftol.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
  ftol.default_value = 1e-6;
  space.add_descriptor(ftol);

  ParameterDescriptor xtol;
  xtol.name = "xtol";
  xtol.type = ParameterType::Continuous;
  xtol.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
  xtol.default_value = 1e-6;
  space.add_descriptor(xtol);

  ParameterDescriptor memory;
  memory.name = "memory";
  memory.type = ParameterType::Boolean;
  memory.default_value = false;
  space.add_descriptor(memory);

  ParameterDescriptor force_bounds;
  force_bounds.name = "force_bounds";
  force_bounds.type = ParameterType::Boolean;
  force_bounds.default_value = false;
  space.add_descriptor(force_bounds);

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

PagmoCmaesHyperOptimizer::PagmoCmaesHyperOptimizer()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoCmaesHyperOptimizer::configure(const ParameterSet &parameters) {
  configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

core::HyperparameterOptimizationResult PagmoCmaesHyperOptimizer::optimize(
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

    const auto seed32 =
        static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());

    pagmo::algorithm algorithm{pagmo::cmaes{
        generations, std::get<double>(configured_parameters_.at("cc")),
        std::get<double>(configured_parameters_.at("cs")),
        std::get<double>(configured_parameters_.at("c1")),
        std::get<double>(configured_parameters_.at("cmu")),
        std::get<double>(configured_parameters_.at("sigma0")),
        std::get<double>(configured_parameters_.at("ftol")),
        std::get<double>(configured_parameters_.at("xtol")),
        std::get<bool>(configured_parameters_.at("memory")),
        std::get<bool>(configured_parameters_.at("force_bounds")), seed32}};

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
