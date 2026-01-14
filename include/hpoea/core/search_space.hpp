#pragma once

#include "hpoea/core/parameters.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace hpoea::core {

enum class SearchMode {
  optimize,
  fixed,
  exclude
};

enum class Transform {
  none,
  log,
  log2,
  sqrt
};

struct ParameterConfig {
  SearchMode mode{SearchMode::optimize};
  std::optional<ParameterValue> fixed_value;
  std::optional<ContinuousRange> continuous_bounds;
  std::optional<IntegerRange> integer_bounds;
  std::vector<ParameterValue> discrete_choices;
  Transform transform{Transform::none};
};

struct EffectiveBounds {
  std::string name;
  ParameterType type;
  SearchMode mode;
  std::optional<ContinuousRange> continuous_bounds;
  std::optional<IntegerRange> integer_bounds;
  std::size_t discrete_choice_count{0};
  Transform transform{Transform::none};
};

class SearchSpace {
public:
  SearchSpace() = default;

  void set(const std::string &name, ParameterConfig config);
  void fix(const std::string &name, ParameterValue value);
  void exclude(const std::string &name);
  void optimize(const std::string &name, ContinuousRange bounds,
                Transform transform = Transform::none);
  void optimize(const std::string &name, IntegerRange bounds);
  void optimize_choices(const std::string &name,
                        std::vector<ParameterValue> choices);

  [[nodiscard]] const ParameterConfig *get(const std::string &name) const;
  [[nodiscard]] bool has(const std::string &name) const;
  [[nodiscard]] const std::unordered_map<std::string, ParameterConfig> &
  configs() const noexcept;
  [[nodiscard]] bool empty() const noexcept;

  void validate(const ParameterSpace &space) const;
  void validate_and_clamp(const ParameterSpace &space);

  [[nodiscard]] std::vector<EffectiveBounds>
  get_effective_bounds(const ParameterSpace &space) const;

  [[nodiscard]] std::size_t
  get_optimization_dimension(const ParameterSpace &space) const;

private:
  std::unordered_map<std::string, ParameterConfig> configs_;
};

[[nodiscard]] double apply_transform(double value, Transform transform);
[[nodiscard]] ContinuousRange transform_bounds(ContinuousRange bounds,
                                               Transform transform);
void validate_transform_bounds(ContinuousRange bounds, Transform transform);
[[nodiscard]] ContinuousRange clamp_bounds(ContinuousRange custom,
                                           ContinuousRange constraint);
[[nodiscard]] IntegerRange clamp_bounds(IntegerRange custom,
                                        IntegerRange constraint);

} // namespace hpoea::core
