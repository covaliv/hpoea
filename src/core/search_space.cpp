#include "hpoea/core/search_space.hpp"

#include <cmath>
#include <stdexcept>

namespace hpoea::core {

void SearchSpace::set(const std::string &name, ParameterConfig config) {
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
  ParameterConfig config;
  config.mode = SearchMode::optimize;
  config.continuous_bounds = bounds;
  config.transform = transform;
  configs_[name] = std::move(config);
}

void SearchSpace::optimize(const std::string &name, IntegerRange bounds) {
  ParameterConfig config;
  config.mode = SearchMode::optimize;
  config.integer_bounds = bounds;
  configs_[name] = std::move(config);
}

void SearchSpace::optimize_choices(const std::string &name,
                                   std::vector<ParameterValue> choices) {
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
  }
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
