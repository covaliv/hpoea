#include "hpoea/core/search_space.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hpoea::core {

void SearchSpace::set(const std::string &name, ParameterConfig config) {
  if (config.continuous_bounds.has_value()) {
    validate_transform_bounds(config.continuous_bounds.value(),
                              config.transform);
  }
  configs_[name] = std::move(config);
}

void SearchSpace::fix(const std::string &name, ParameterValue value) {
  ParameterConfig config;
  config.mode = SearchMode::fixed;
  config.fixed_value = std::move(value);
  configs_[name] = std::move(config);
}

void SearchSpace::exclude(const std::string &name) {
  ParameterConfig config;
  config.mode = SearchMode::exclude;
  configs_[name] = std::move(config);
}

void SearchSpace::optimize(const std::string &name, ContinuousRange bounds,
                           Transform transform) {
  validate_transform_bounds(bounds, transform);
  ParameterConfig config;
  config.mode = SearchMode::optimize;
  config.continuous_bounds = bounds;
  config.transform = transform;
  configs_[name] = std::move(config);
}

void SearchSpace::optimize(const std::string &name, IntegerRange bounds) {
  if (bounds.lower > bounds.upper) {
    throw ParameterValidationError("invalid integer bounds for '" + name +
                                   "': lower > upper");
  }
  ParameterConfig config;
  config.mode = SearchMode::optimize;
  config.integer_bounds = bounds;
  configs_[name] = std::move(config);
}

void SearchSpace::optimize_choices(const std::string &name,
                                   std::vector<ParameterValue> choices) {
  if (choices.empty()) {
    throw ParameterValidationError("discrete choices for '" + name +
                                   "' cannot be empty");
  }
  ParameterConfig config;
  config.mode = SearchMode::optimize;
  config.discrete_choices = std::move(choices);
  configs_[name] = std::move(config);
}

const ParameterConfig *SearchSpace::get(const std::string &name) const {
  auto it = configs_.find(name);
  return it != configs_.end() ? &it->second : nullptr;
}

bool SearchSpace::has(const std::string &name) const {
  return configs_.find(name) != configs_.end();
}

const std::unordered_map<std::string, ParameterConfig> &
SearchSpace::configs() const noexcept {
  return configs_;
}

bool SearchSpace::empty() const noexcept { return configs_.empty(); }

void SearchSpace::validate(const ParameterSpace &space) const {
  for (const auto &[name, config] : configs_) {
    if (!space.contains(name)) {
      throw ParameterValidationError("search space references unknown "
                                     "parameter: " +
                                     name);
    }

    const auto &descriptor = space.descriptor(name);

    if (config.mode == SearchMode::fixed && config.fixed_value.has_value()) {
      const auto &value = config.fixed_value.value();
      if (descriptor.type == ParameterType::Continuous) {
        if (!std::holds_alternative<double>(value)) {
          throw ParameterValidationError("fixed value for '" + name +
                                         "' must be double");
        }
        if (descriptor.continuous_range.has_value()) {
          double v = std::get<double>(value);
          if (v < descriptor.continuous_range->lower ||
              v > descriptor.continuous_range->upper) {
            throw ParameterValidationError("fixed value for '" + name +
                                           "' outside valid range");
          }
        }
      } else if (descriptor.type == ParameterType::Integer) {
        if (!std::holds_alternative<std::int64_t>(value)) {
          throw ParameterValidationError("fixed value for '" + name +
                                         "' must be integer");
        }
        if (descriptor.integer_range.has_value()) {
          auto v = std::get<std::int64_t>(value);
          if (v < descriptor.integer_range->lower ||
              v > descriptor.integer_range->upper) {
            throw ParameterValidationError("fixed value for '" + name +
                                           "' outside valid range");
          }
        }
      }
    }

    if (config.continuous_bounds.has_value() &&
        descriptor.type != ParameterType::Continuous) {
      throw ParameterValidationError("continuous bounds specified for "
                                     "non-continuous parameter: " +
                                     name);
    }

    if (config.integer_bounds.has_value() &&
        descriptor.type != ParameterType::Integer) {
      throw ParameterValidationError("integer bounds specified for "
                                     "non-integer parameter: " +
                                     name);
    }
  }
}

void SearchSpace::validate_and_clamp(const ParameterSpace &space) {
  validate(space);

  for (auto &[name, config] : configs_) {
    if (config.mode != SearchMode::optimize) {
      continue;
    }

    const auto &descriptor = space.descriptor(name);

    if (config.continuous_bounds.has_value() &&
        descriptor.continuous_range.has_value()) {
      config.continuous_bounds =
          clamp_bounds(config.continuous_bounds.value(),
                       descriptor.continuous_range.value());
      validate_transform_bounds(config.continuous_bounds.value(),
                                config.transform);
    }

    if (config.integer_bounds.has_value() &&
        descriptor.integer_range.has_value()) {
      config.integer_bounds = clamp_bounds(config.integer_bounds.value(),
                                           descriptor.integer_range.value());
      if (config.integer_bounds->lower > config.integer_bounds->upper) {
        throw ParameterValidationError(
            "integer bounds for '" + name +
            "' do not overlap with parameter range");
      }
    }
  }
}

std::vector<EffectiveBounds>
SearchSpace::get_effective_bounds(const ParameterSpace &space) const {
  std::vector<EffectiveBounds> result;

  for (const auto &descriptor : space.descriptors()) {
    EffectiveBounds eb;
    eb.name = descriptor.name;
    eb.type = descriptor.type;

    const auto *config = get(descriptor.name);

    if (config) {
      eb.mode = config->mode;
      eb.transform = config->transform;

      if (config->mode == SearchMode::optimize) {
        if (!config->discrete_choices.empty()) {
          eb.discrete_choice_count = config->discrete_choices.size();
        } else if (config->continuous_bounds.has_value()) {
          eb.continuous_bounds = config->continuous_bounds;
        } else if (config->integer_bounds.has_value()) {
          eb.integer_bounds = config->integer_bounds;
        } else {
          eb.continuous_bounds = descriptor.continuous_range;
          eb.integer_bounds = descriptor.integer_range;
        }
      }
    } else {
      eb.mode = SearchMode::optimize;
      eb.continuous_bounds = descriptor.continuous_range;
      eb.integer_bounds = descriptor.integer_range;
    }

    result.push_back(eb);
  }

  return result;
}

std::size_t
SearchSpace::get_optimization_dimension(const ParameterSpace &space) const {
  std::size_t dim = 0;

  for (const auto &descriptor : space.descriptors()) {
    const auto *config = get(descriptor.name);

    if (config) {
      if (config->mode == SearchMode::fixed ||
          config->mode == SearchMode::exclude) {
        continue;
      }
    }

    ++dim;
  }

  return dim;
}

void validate_transform_bounds(ContinuousRange bounds, Transform transform) {
  if (bounds.lower > bounds.upper) {
    throw ParameterValidationError("invalid bounds: lower > upper");
  }

  switch (transform) {
  case Transform::none:
    break;
  case Transform::log:
  case Transform::log2:
    if (bounds.lower <= 0.0) {
      throw ParameterValidationError(
          "log transform requires positive bounds, got lower=" +
          std::to_string(bounds.lower));
    }
    break;
  case Transform::sqrt:
    if (bounds.lower < 0.0) {
      throw ParameterValidationError(
          "sqrt transform requires non-negative bounds, got lower=" +
          std::to_string(bounds.lower));
    }
    break;
  }
}

ContinuousRange clamp_bounds(ContinuousRange custom, ContinuousRange constraint) {
  return {std::max(custom.lower, constraint.lower),
          std::min(custom.upper, constraint.upper)};
}

IntegerRange clamp_bounds(IntegerRange custom, IntegerRange constraint) {
  return {std::max(custom.lower, constraint.lower),
          std::min(custom.upper, constraint.upper)};
}

double apply_transform(double value, Transform transform) {
  switch (transform) {
  case Transform::none:
    return value;
  case Transform::log:
    return std::pow(10.0, value);
  case Transform::log2:
    return std::pow(2.0, value);
  case Transform::sqrt:
    return value * value;
  }
  return value;
}

ContinuousRange transform_bounds(ContinuousRange bounds, Transform transform) {
  validate_transform_bounds(bounds, transform);
  switch (transform) {
  case Transform::none:
    return bounds;
  case Transform::log:
    return {std::log10(bounds.lower), std::log10(bounds.upper)};
  case Transform::log2:
    return {std::log2(bounds.lower), std::log2(bounds.upper)};
  case Transform::sqrt:
    return {std::sqrt(bounds.lower), std::sqrt(bounds.upper)};
  }
  return bounds;
}

} // namespace hpoea::core
