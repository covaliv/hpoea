#include "hpoea/wrappers/pagmo/nm_hyper.hpp"

#include "hyper_tuning_udp.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/nlopt.hpp>
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

constexpr std::string_view kFamily = "NelderMead";
constexpr std::string_view kImplementation = "nlopt::neldermead";
constexpr std::string_view kVersion = "2.x";

ParameterSpace make_parameter_space() {
  ParameterSpace space;

  ParameterDescriptor max_fevals;
  max_fevals.name = "max_fevals";
  max_fevals.type = ParameterType::Integer;
  max_fevals.integer_range = hpoea::core::IntegerRange{1, 100000};
  max_fevals.default_value = static_cast<std::int64_t>(1000);
  space.add_descriptor(max_fevals);

  ParameterDescriptor xtol_rel;
  xtol_rel.name = "xtol_rel";
  xtol_rel.type = ParameterType::Continuous;
  xtol_rel.continuous_range = hpoea::core::ContinuousRange{1e-15, 1e-1};
  xtol_rel.default_value = 1e-8;
  space.add_descriptor(xtol_rel);

  ParameterDescriptor ftol_rel;
  ftol_rel.name = "ftol_rel";
  ftol_rel.type = ParameterType::Continuous;
  ftol_rel.continuous_range = hpoea::core::ContinuousRange{1e-15, 1e-1};
  ftol_rel.default_value = 1e-8;
  space.add_descriptor(ftol_rel);

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

PagmoNelderMeadHyperOptimizer::PagmoNelderMeadHyperOptimizer()
    : parameter_space_(make_parameter_space()),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(make_identity()) {}

void PagmoNelderMeadHyperOptimizer::configure(const ParameterSet &parameters) {
  configured_parameters_ = parameter_space_.apply_defaults(parameters);
}

core::HyperparameterOptimizationResult PagmoNelderMeadHyperOptimizer::optimize(
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

    auto max_fevals = static_cast<unsigned>(
        std::get<std::int64_t>(configured_parameters_.at("max_fevals")));
    if (budget.function_evaluations.has_value()) {
      max_fevals = std::min<unsigned>(
          max_fevals,
          static_cast<unsigned>(budget.function_evaluations.value()));
    }

    const auto xtol_rel =
        std::get<double>(configured_parameters_.at("xtol_rel"));
    const auto ftol_rel =
        std::get<double>(configured_parameters_.at("ftol_rel"));

    const auto seed32 =
        static_cast<unsigned>(seed & std::numeric_limits<unsigned>::max());

    pagmo::nlopt nm_alg("neldermead");
    nm_alg.set_maxeval(static_cast<int>(max_fevals));
    nm_alg.set_xtol_rel(xtol_rel);
    nm_alg.set_ftol_rel(ftol_rel);
    pagmo::algorithm algorithm{nm_alg};

    const auto dimension = lower.size();
    const auto population_size =
        static_cast<pagmo::population::size_type>(dimension + 1);

    pagmo::population population{tuning_problem, population_size, seed32};

    const auto start_time = std::chrono::steady_clock::now();
    population = algorithm.evolve(population);
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
    result.budget_usage.generations = 1;
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
