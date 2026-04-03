#pragma once

#include "budget_util.hpp"
#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/problem.hpp"
#include "hpoea/core/search_space.hpp"
#include "hpoea/core/types.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <pagmo/types.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace hpoea::pagmo_wrappers {

struct HyperparameterTuningProblem {
  struct Context {
    const core::IEvolutionaryAlgorithmFactory *factory{nullptr};
    const core::IProblem *problem{nullptr};
    core::Budget algorithm_budget;
    unsigned long base_seed{0};
    std::shared_ptr<std::vector<core::HyperparameterTrialRecord>> trials;
    std::shared_ptr<core::SearchSpace> search_space;
    mutable std::optional<core::HyperparameterTrialRecord> best_trial;
    mutable std::atomic<std::size_t> evaluations{0};
    mutable std::mutex mutex;

    [[nodiscard]] std::optional<core::HyperparameterTrialRecord>
    get_best_trial() const {
      std::scoped_lock lock(mutex);
      return best_trial;
    }

    [[nodiscard]] std::size_t get_evaluations() const {
      return evaluations.load(std::memory_order_relaxed);
    }
  };

  HyperparameterTuningProblem() = default;

  explicit HyperparameterTuningProblem(std::shared_ptr<Context> context)
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
        auto [range, transform] = resolve_continuous(descriptor, config);
        const auto transformed = core::transform_bounds(range, transform);
        lower.push_back(transformed.lower);
        upper.push_back(transformed.upper);
        break;
      }
      case core::ParameterType::Integer: {
        if (config && !config->discrete_choices.empty()) {
          lower.push_back(0.0);
          upper.push_back(static_cast<double>(config->discrete_choices.size() - 1));
        } else {
          auto range = resolve_integer_range(descriptor, config);
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
        const auto count = (config && !config->discrete_choices.empty())
            ? config->discrete_choices.size()
            : descriptor.categorical_choices.size();
        if (count == 0) {
          throw core::ParameterValidationError(
              "categorical parameter '" + descriptor.name + "' has zero choices");
        }
        lower.push_back(0.0);
        upper.push_back(static_cast<double>(count - 1));
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
    const auto expected_dim = compute_candidate_dimension(ctx, descriptors);

    if (candidate.size() != expected_dim) {
      throw std::invalid_argument(
          "HyperparameterTuningProblem candidate dimension mismatch: expected " +
          std::to_string(expected_dim) + ", got " +
          std::to_string(candidate.size()));
    }

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
      case core::ParameterType::Continuous:
        parameters.emplace(descriptor.name, decode_continuous(value, descriptor, config));
        break;
      case core::ParameterType::Integer:
        parameters.emplace(descriptor.name, decode_integer(value, descriptor, config));
        break;
      case core::ParameterType::Boolean:
        parameters.emplace(descriptor.name, value > 0.5);
        break;
      case core::ParameterType::Categorical:
        parameters.emplace(descriptor.name, decode_categorical(value, descriptor, config));
        break;
      }
    }

    // apply defaults only for non-excluded parameters.
    for (const auto &desc : descriptors) {
      if (parameters.contains(desc.name)) continue;
      const core::ParameterConfig *pc = nullptr;
      if (ctx.search_space) pc = ctx.search_space->get(desc.name);
      if (pc && pc->mode == core::SearchMode::exclude) continue;
      if (desc.default_value.has_value()) {
        parameters.emplace(desc.name, *desc.default_value);
      }
    }

    auto algorithm = ctx.factory->create();
    algorithm->configure(parameters);
    const auto eval_index =
      ctx.evaluations.fetch_add(1, std::memory_order_relaxed);
    const unsigned long eval_seed =
      derive_seed(ctx.base_seed, static_cast<unsigned long>(eval_index));

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
      const bool candidate_finite =
          std::isfinite(record.optimization_result.best_fitness);
      const bool current_finite = ctx.best_trial &&
          std::isfinite(ctx.best_trial->optimization_result.best_fitness);
      const bool should_update_best =
          !ctx.best_trial ||
          (candidate_finite && (!current_finite ||
              record.optimization_result.best_fitness <
                  ctx.best_trial->optimization_result.best_fitness));
      if (should_update_best) {
        ctx.best_trial = record;
      }
    }

    constexpr double FAILED_TRIAL_PENALTY = 1e20;
    const auto fitness = std::isfinite(record.optimization_result.best_fitness)
        ? record.optimization_result.best_fitness
        : FAILED_TRIAL_PENALTY;
    return pagmo::vector_double{fitness};
  }

  [[nodiscard]] bool has_gradient() const { return false; }

  [[nodiscard]] bool has_hessians() const { return false; }

  [[nodiscard]] std::string get_name() const { return "HyperparameterTuningProblem"; }

private:
  struct ResolvedContinuous {
    core::ContinuousRange range;
    core::Transform transform;
  };

  [[nodiscard]] static ResolvedContinuous resolve_continuous(
      const core::ParameterDescriptor &desc,
      const core::ParameterConfig *config) {
    if (!desc.continuous_range.has_value()) {
      throw std::logic_error("continuous parameter missing range: " + desc.name);
    }
    auto range = *desc.continuous_range;
    auto transform = core::Transform::none;
    if (config) {
      if (config->continuous_bounds.has_value()) {
        range = config->continuous_bounds.value();
      }
      transform = config->transform;
    }
    return {range, transform};
  }

  [[nodiscard]] static core::IntegerRange resolve_integer_range(
      const core::ParameterDescriptor &desc,
      const core::ParameterConfig *config) {
    if (!desc.integer_range.has_value()) {
      throw std::logic_error("integer parameter missing range: " + desc.name);
    }
    auto range = *desc.integer_range;
    if (config && config->integer_bounds.has_value()) {
      range = config->integer_bounds.value();
    }
    return range;
  }

  [[nodiscard]] static core::ParameterValue decode_discrete_choice(
      double value, const std::vector<core::ParameterValue> &choices) {
    auto index = static_cast<std::size_t>(
        std::clamp<std::int64_t>(std::llround(value), 0,
                                 static_cast<std::int64_t>(choices.size() - 1)));
    return choices[index];
  }

  [[nodiscard]] static core::ParameterValue decode_continuous(
      double value, const core::ParameterDescriptor &desc,
      const core::ParameterConfig *config) {
    auto [range, transform] = resolve_continuous(desc, config);
    double numeric = core::inverse_transform(value, transform);
    return std::clamp(numeric, range.lower, range.upper);
  }

  [[nodiscard]] static core::ParameterValue decode_integer(
      double value, const core::ParameterDescriptor &desc,
      const core::ParameterConfig *config) {
    if (config && !config->discrete_choices.empty()) {
      return decode_discrete_choice(value, config->discrete_choices);
    }
    auto range = resolve_integer_range(desc, config);
    auto rounded = static_cast<std::int64_t>(std::llround(value));
    return std::clamp(rounded, range.lower, range.upper);
  }

  [[nodiscard]] static core::ParameterValue decode_categorical(
      double value, const core::ParameterDescriptor &desc,
      const core::ParameterConfig *config) {
    if (config && !config->discrete_choices.empty()) {
      return decode_discrete_choice(value, config->discrete_choices);
    }
    const auto &choices = desc.categorical_choices;
    if (choices.empty()) {
      throw core::ParameterValidationError(
          "Categorical descriptor without choices: " + desc.name);
    }
    auto index = static_cast<std::size_t>(
        std::clamp<std::int64_t>(std::llround(value), 0,
                                 static_cast<std::int64_t>(choices.size() - 1)));
    return std::string{choices[index]};
  }

  [[nodiscard]] static std::size_t compute_candidate_dimension(
      const Context &ctx,
      const std::vector<core::ParameterDescriptor> &descriptors) {
    if (ctx.search_space) {
      return ctx.search_space->get_optimization_dimension(
          ctx.factory->parameter_space());
    }
    return descriptors.size();
  }

  [[nodiscard]] const Context &ensure_context() const {
    if (!context_) {
      throw std::runtime_error(
          "HyperparameterTuningProblem used without associated context");
    }
    return *context_;
  }

  std::shared_ptr<Context> context_;
};

} // namespace hpoea::pagmo_wrappers
