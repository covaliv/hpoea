#pragma once

#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/search_space.hpp"
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
    std::shared_ptr<core::SearchSpace> search_space;
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
      const core::ParameterConfig *config = nullptr;
      if (ctx.search_space) {
        config = ctx.search_space->get(descriptor.name);
      }

      if (config) {
        if (config->mode == core::SearchMode::fixed ||
            config->mode == core::SearchMode::exclude) {
          continue;
        }
      }

      switch (descriptor.type) {
      case core::ParameterType::Continuous: {
        auto range = descriptor.continuous_range.value_or(
            core::ContinuousRange{-1.0, 1.0});
        core::Transform transform = core::Transform::none;

        if (config) {
          if (config->continuous_bounds.has_value()) {
            range = config->continuous_bounds.value();
          }
          transform = config->transform;
        }

        const auto transformed = core::transform_bounds(range, transform);
        lower.push_back(transformed.lower);
        upper.push_back(transformed.upper);
        break;
      }
      case core::ParameterType::Integer: {
        if (config && !config->discrete_choices.empty()) {
          lower.push_back(0.0);
          upper.push_back(
              static_cast<double>(config->discrete_choices.size() - 1));
        } else {
          auto range =
              descriptor.integer_range.value_or(core::IntegerRange{-100, 100});

          if (config && config->integer_bounds.has_value()) {
            range = config->integer_bounds.value();
          }

          lower.push_back(static_cast<double>(range.lower));
          upper.push_back(static_cast<double>(range.upper));
        }
        break;
      }
      case core::ParameterType::Boolean: {
        lower.push_back(0.0);
        upper.push_back(1.0);
        break;
      }
      case core::ParameterType::Categorical: {
        if (config && !config->discrete_choices.empty()) {
          lower.push_back(0.0);
          upper.push_back(
              static_cast<double>(config->discrete_choices.size() - 1));
        } else {
          lower.push_back(0.0);
          upper.push_back(
              static_cast<double>(descriptor.categorical_choices.size() - 1));
        }
        break;
      }
      }
    }

    if (lower.empty()) {
      throw core::ParameterValidationError(
          "All parameters are fixed or excluded. At least one parameter "
          "must be optimized.");
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

    std::size_t candidate_index = 0;

    for (const auto &descriptor : descriptors) {
      const core::ParameterConfig *config = nullptr;
      if (ctx.search_space) {
        config = ctx.search_space->get(descriptor.name);
      }

      if (config) {
        if (config->mode == core::SearchMode::fixed) {
          if (config->fixed_value.has_value()) {
            parameters.emplace(descriptor.name, config->fixed_value.value());
          }
          continue;
        }
        if (config->mode == core::SearchMode::exclude) {
          continue;
        }
      }

      const auto value = candidate[candidate_index++];

      switch (descriptor.type) {
      case core::ParameterType::Continuous: {
        auto range = descriptor.continuous_range.value_or(
            core::ContinuousRange{-1e10, 1e10});
        core::Transform transform = core::Transform::none;

        if (config) {
          if (config->continuous_bounds.has_value()) {
            range = config->continuous_bounds.value();
          }
          transform = config->transform;
        }

        double numeric = core::apply_transform(value, transform);
        numeric = std::clamp(numeric, range.lower, range.upper);
        parameters.emplace(descriptor.name, numeric);
        break;
      }
      case core::ParameterType::Integer: {
        if (config && !config->discrete_choices.empty()) {
          const auto &choices = config->discrete_choices;
          auto index_value = static_cast<std::size_t>(
              std::clamp<std::int64_t>(std::llround(value), 0,
                                       static_cast<std::int64_t>(choices.size() - 1)));
          parameters.emplace(descriptor.name, choices[index_value]);
        } else {
          auto range =
              descriptor.integer_range.value_or(core::IntegerRange{-100, 100});

          if (config && config->integer_bounds.has_value()) {
            range = config->integer_bounds.value();
          }

          auto rounded = static_cast<std::int64_t>(std::llround(value));
          rounded = std::clamp(rounded, range.lower, range.upper);
          parameters.emplace(descriptor.name, rounded);
        }
        break;
      }
      case core::ParameterType::Boolean: {
        parameters.emplace(descriptor.name, value > 0.5);
        break;
      }
      case core::ParameterType::Categorical: {
        if (config && !config->discrete_choices.empty()) {
          const auto &choices = config->discrete_choices;
          auto index_value = static_cast<std::size_t>(
              std::clamp<std::int64_t>(std::llround(value), 0,
                                       static_cast<std::int64_t>(choices.size() - 1)));
          parameters.emplace(descriptor.name, choices[index_value]);
        } else {
          const auto &choices = descriptor.categorical_choices;
          if (choices.empty()) {
            throw core::ParameterValidationError(
                "Categorical descriptor without choices: " + descriptor.name);
          }
          auto index_value = static_cast<std::size_t>(
              std::clamp<std::int64_t>(std::llround(value), 0,
                                       static_cast<std::int64_t>(choices.size() - 1)));
          parameters.emplace(descriptor.name, choices[index_value]);
        }
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
