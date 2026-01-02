#pragma once

#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <pagmo/types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpoea::pagmo_wrappers {

struct HyperTuningUdp {
  struct Context {
    const core::IEvolutionaryAlgorithmFactory *factory{nullptr};
    const core::IProblem *problem{nullptr};
    core::Budget algorithm_budget;
    unsigned long base_seed{0};
    std::shared_ptr<std::vector<core::HyperparameterTrialRecord>> trials;
    mutable std::optional<core::HyperparameterTrialRecord> best_trial;
    mutable std::size_t evaluations{0};
    mutable std::mutex mutex;

    [[nodiscard]] std::optional<core::HyperparameterTrialRecord>
    get_best_trial() const {
      std::scoped_lock lock(mutex);
      return best_trial;
    }

    [[nodiscard]] std::size_t get_evaluations() const {
      std::scoped_lock lock(mutex);
      return evaluations;
    }
  };

  HyperTuningUdp() = default;

  explicit HyperTuningUdp(std::shared_ptr<Context> context)
      : context_(std::move(context)) {}

  [[nodiscard]] std::pair<pagmo::vector_double, pagmo::vector_double>
  get_bounds() const {
    const auto &ctx = ensure_context();
    const auto &space = ctx.factory->parameter_space();
    const auto &descriptors = space.descriptors();
    if (descriptors.empty()) {
      throw core::ParameterValidationError(
          "Algorithm parameter space is empty. Hyperparameter optimizer "
          "requires at least one parameter.");
    }

    pagmo::vector_double lower;
    pagmo::vector_double upper;
    lower.reserve(descriptors.size());
    upper.reserve(descriptors.size());

    for (const auto &descriptor : descriptors) {
      switch (descriptor.type) {
      case core::ParameterType::Continuous: {
        const auto range = descriptor.continuous_range.value_or(
            core::ContinuousRange{-1.0, 1.0});
        lower.push_back(range.lower);
        upper.push_back(range.upper);
        break;
      }
      case core::ParameterType::Integer: {
        const auto range =
            descriptor.integer_range.value_or(core::IntegerRange{-100, 100});
        lower.push_back(static_cast<double>(range.lower));
        upper.push_back(static_cast<double>(range.upper));
        break;
      }
      case core::ParameterType::Boolean: {
        lower.push_back(0.0);
        upper.push_back(1.0);
        break;
      }
      case core::ParameterType::Categorical: {
        lower.push_back(0.0);
        upper.push_back(
            static_cast<double>(descriptor.categorical_choices.size() - 1));
        break;
      }
      }
    }

    return {lower, upper};
  }

  [[nodiscard]] pagmo::vector_double
  fitness(const pagmo::vector_double &candidate) const {
    const auto &ctx = ensure_context();
    const auto &space = ctx.factory->parameter_space();
    const auto &descriptors = space.descriptors();

    core::ParameterSet parameters;
    parameters.reserve(descriptors.size());

    for (std::size_t index = 0; index < descriptors.size(); ++index) {
      const auto &descriptor = descriptors[index];
      const auto value = candidate[index];

      switch (descriptor.type) {
      case core::ParameterType::Continuous: {
        double numeric = value;
        if (descriptor.continuous_range.has_value()) {
          numeric = std::clamp(numeric, descriptor.continuous_range->lower,
                               descriptor.continuous_range->upper);
        }
        parameters.emplace(descriptor.name, numeric);
        break;
      }
      case core::ParameterType::Integer: {
        auto rounded = static_cast<std::int64_t>(std::llround(value));
        if (descriptor.integer_range.has_value()) {
          rounded = std::clamp(rounded, descriptor.integer_range->lower,
                               descriptor.integer_range->upper);
        }
        parameters.emplace(descriptor.name, rounded);
        break;
      }
      case core::ParameterType::Boolean: {
        parameters.emplace(descriptor.name, value > 0.5);
        break;
      }
      case core::ParameterType::Categorical: {
        const auto &choices = descriptor.categorical_choices;
        if (choices.empty()) {
          throw core::ParameterValidationError(
              "Categorical descriptor without choices: " + descriptor.name);
        }
        auto index_value = static_cast<std::int64_t>(std::llround(value));
        index_value = std::clamp<std::int64_t>(
            index_value, 0, static_cast<std::int64_t>(choices.size() - 1));
        parameters.emplace(descriptor.name,
                           choices[static_cast<std::size_t>(index_value)]);
        break;
      }
      }
    }

    parameters = space.apply_defaults(parameters);

    auto algorithm = ctx.factory->create();
    algorithm->configure(parameters);
    unsigned long eval_seed;
    {
      std::scoped_lock lock(ctx.mutex);
      eval_seed = ctx.base_seed + static_cast<unsigned long>(ctx.evaluations++);
    }

    const auto result =
        algorithm->run(*ctx.problem, ctx.algorithm_budget, eval_seed);

    core::HyperparameterTrialRecord record;
    record.parameters = parameters;
    record.optimization_result = result;

    {
      std::scoped_lock lock(ctx.mutex);
      if (ctx.trials) {
        ctx.trials->push_back(record);
      }
      const bool should_update_best =
          !ctx.best_trial ||
          record.optimization_result.best_fitness <
              ctx.best_trial->optimization_result.best_fitness;
      if (should_update_best) {
        ctx.best_trial = record;
      }
    }

    return pagmo::vector_double{record.optimization_result.best_fitness};
  }

  [[nodiscard]] bool has_gradient() const { return false; }

  [[nodiscard]] bool has_hessians() const { return false; }

  [[nodiscard]] std::string get_name() const { return "HyperTuningUdp"; }

private:
  [[nodiscard]] const Context &ensure_context() const {
    if (!context_) {
      throw std::runtime_error(
          "HyperTuningUdp used without associated context");
    }
    return *context_;
  }

  std::shared_ptr<Context> context_;
};

} // namespace hpoea::pagmo_wrappers
