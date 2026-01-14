#include "hpoea/core/parameters.hpp"
#include "hpoea/core/search_space.hpp"
#include "hpoea/core/types.hpp"
#include "hpoea/wrappers/pagmo/cmaes_hyper.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string_view>

using namespace hpoea;

namespace {

bool test_search_space_basic() {
  std::cout << "  basic operations\n";

  core::SearchSpace space;

  space.fix("param_a", 42.0);
  space.exclude("param_b");
  space.optimize("param_c", core::ContinuousRange{0.1, 10.0});
  space.optimize("param_d", core::IntegerRange{1, 100});
  space.optimize_choices("param_e",
                         {std::int64_t{1}, std::int64_t{2}, std::int64_t{5}});

  if (space.empty()) {
    std::cerr << "    expected non-empty space\n";
    return false;
  }

  const auto *config_a = space.get("param_a");
  if (!config_a || config_a->mode != core::SearchMode::fixed) {
    std::cerr << "    param_a should be fixed\n";
    return false;
  }

  const auto *config_b = space.get("param_b");
  if (!config_b || config_b->mode != core::SearchMode::exclude) {
    std::cerr << "    param_b should be excluded\n";
    return false;
  }

  const auto *config_c = space.get("param_c");
  if (!config_c || config_c->mode != core::SearchMode::optimize) {
    std::cerr << "    param_c should be optimized\n";
    return false;
  }
  if (!config_c->continuous_bounds ||
      config_c->continuous_bounds->lower != 0.1 ||
      config_c->continuous_bounds->upper != 10.0) {
    std::cerr << "    param_c has wrong bounds\n";
    return false;
  }

  const auto *config_d = space.get("param_d");
  if (!config_d || !config_d->integer_bounds ||
      config_d->integer_bounds->lower != 1 ||
      config_d->integer_bounds->upper != 100) {
    std::cerr << "    param_d has wrong integer bounds\n";
    return false;
  }

  const auto *config_e = space.get("param_e");
  if (!config_e || config_e->discrete_choices.size() != 3) {
    std::cerr << "    param_e has wrong discrete choices\n";
    return false;
  }

  if (space.has("nonexistent")) {
    std::cerr << "    has() returned true for nonexistent param\n";
    return false;
  }

  return true;
}

bool test_transforms() {
  std::cout << "  transform functions\n";

  constexpr double tolerance = 1e-9;

  // log transform: 10^value
  double v = core::apply_transform(2.0, core::Transform::log);
  if (std::abs(v - 100.0) > tolerance) {
    std::cerr << "    log transform failed: " << v << " != 100.0\n";
    return false;
  }

  // log2 transform: 2^value
  v = core::apply_transform(3.0, core::Transform::log2);
  if (std::abs(v - 8.0) > tolerance) {
    std::cerr << "    log2 transform failed: " << v << " != 8.0\n";
    return false;
  }

  // sqrt transform: value^2
  v = core::apply_transform(3.0, core::Transform::sqrt);
  if (std::abs(v - 9.0) > tolerance) {
    std::cerr << "    sqrt transform failed: " << v << " != 9.0\n";
    return false;
  }

  // transform_bounds for log: [0.01, 100] -> [-2, 2]
  auto bounds = core::transform_bounds({0.01, 100.0}, core::Transform::log);
  if (std::abs(bounds.lower - (-2.0)) > tolerance ||
      std::abs(bounds.upper - 2.0) > tolerance) {
    std::cerr << "    transform_bounds(log) failed: [" << bounds.lower << ", "
              << bounds.upper << "]\n";
    return false;
  }

  // transform_bounds for log2: [1, 8] -> [0, 3]
  bounds = core::transform_bounds({1.0, 8.0}, core::Transform::log2);
  if (std::abs(bounds.lower) > tolerance ||
      std::abs(bounds.upper - 3.0) > tolerance) {
    std::cerr << "    transform_bounds(log2) failed: [" << bounds.lower << ", "
              << bounds.upper << "]\n";
    return false;
  }

  // transform_bounds for sqrt: [4, 16] -> [2, 4]
  bounds = core::transform_bounds({4.0, 16.0}, core::Transform::sqrt);
  if (std::abs(bounds.lower - 2.0) > tolerance ||
      std::abs(bounds.upper - 4.0) > tolerance) {
    std::cerr << "    transform_bounds(sqrt) failed: [" << bounds.lower << ", "
              << bounds.upper << "]\n";
    return false;
  }

  return true;
}

bool test_validation() {
  std::cout << "  validation\n";

  core::ParameterSpace param_space;

  core::ParameterDescriptor desc_a;
  desc_a.name = "param_a";
  desc_a.type = core::ParameterType::Continuous;
  param_space.add_descriptor(desc_a);

  core::ParameterDescriptor desc_b;
  desc_b.name = "param_b";
  desc_b.type = core::ParameterType::Integer;
  param_space.add_descriptor(desc_b);

  core::SearchSpace search;
  search.fix("param_a", 1.0);
  search.optimize("param_b", core::IntegerRange{1, 10});

  try {
    search.validate(param_space);
  } catch (const core::ParameterValidationError &e) {
    std::cerr << "    valid search space threw: " << e.what() << "\n";
    return false;
  }

  core::SearchSpace invalid_search;
  invalid_search.fix("nonexistent", 1.0);

  try {
    invalid_search.validate(param_space);
    std::cerr << "    invalid search space should throw\n";
    return false;
  } catch (const core::ParameterValidationError &) {
    // expected
  }

  return true;
}

bool test_transform_validation() {
  std::cout << "  transform bounds validation\n";

  // log with zero lower bound should throw
  try {
    core::SearchSpace space;
    space.optimize("param", core::ContinuousRange{0.0, 1.0}, core::Transform::log);
    std::cerr << "    log with zero lower bound should throw\n";
    return false;
  } catch (const core::ParameterValidationError &) {
    // expected
  }

  // log with negative lower bound should throw
  try {
    core::SearchSpace space;
    space.optimize("param", core::ContinuousRange{-1.0, 1.0}, core::Transform::log);
    std::cerr << "    log with negative lower bound should throw\n";
    return false;
  } catch (const core::ParameterValidationError &) {
    // expected
  }

  // sqrt with negative lower bound should throw
  try {
    core::SearchSpace space;
    space.optimize("param", core::ContinuousRange{-1.0, 1.0}, core::Transform::sqrt);
    std::cerr << "    sqrt with negative lower bound should throw\n";
    return false;
  } catch (const core::ParameterValidationError &) {
    // expected
  }

  // sqrt with zero is fine
  try {
    core::SearchSpace space;
    space.optimize("param", core::ContinuousRange{0.0, 1.0}, core::Transform::sqrt);
  } catch (const core::ParameterValidationError &e) {
    std::cerr << "    sqrt with zero should be fine: " << e.what() << "\n";
    return false;
  }

  return true;
}

bool test_bounds_validation() {
  std::cout << "  bounds validation\n";

  // lower > upper should throw
  try {
    core::SearchSpace space;
    space.optimize("param", core::ContinuousRange{10.0, 1.0});
    std::cerr << "    lower > upper should throw\n";
    return false;
  } catch (const core::ParameterValidationError &) {
    // expected
  }

  // integer lower > upper should throw
  try {
    core::SearchSpace space;
    space.optimize("param", core::IntegerRange{10, 1});
    std::cerr << "    integer lower > upper should throw\n";
    return false;
  } catch (const core::ParameterValidationError &) {
    // expected
  }

  // empty choices should throw
  try {
    core::SearchSpace space;
    space.optimize_choices("param", {});
    std::cerr << "    empty choices should throw\n";
    return false;
  } catch (const core::ParameterValidationError &) {
    // expected
  }

  return true;
}

bool test_validate_and_clamp() {
  std::cout << "  validate and clamp\n";

  core::ParameterSpace param_space;

  core::ParameterDescriptor desc;
  desc.name = "param";
  desc.type = core::ParameterType::Continuous;
  desc.continuous_range = core::ContinuousRange{0.0, 1.0};
  param_space.add_descriptor(desc);

  core::SearchSpace search;
  search.optimize("param", core::ContinuousRange{-0.5, 1.5});

  search.validate_and_clamp(param_space);

  const auto *config = search.get("param");
  if (!config || !config->continuous_bounds) {
    std::cerr << "    config missing after clamp\n";
    return false;
  }

  if (config->continuous_bounds->lower != 0.0 ||
      config->continuous_bounds->upper != 1.0) {
    std::cerr << "    bounds not clamped correctly: ["
              << config->continuous_bounds->lower << ", "
              << config->continuous_bounds->upper << "]\n";
    return false;
  }

  return true;
}

bool test_effective_bounds() {
  std::cout << "  effective bounds inspection\n";

  core::ParameterSpace param_space;

  core::ParameterDescriptor desc_a;
  desc_a.name = "param_a";
  desc_a.type = core::ParameterType::Continuous;
  desc_a.continuous_range = core::ContinuousRange{0.0, 10.0};
  param_space.add_descriptor(desc_a);

  core::ParameterDescriptor desc_b;
  desc_b.name = "param_b";
  desc_b.type = core::ParameterType::Integer;
  desc_b.integer_range = core::IntegerRange{1, 100};
  param_space.add_descriptor(desc_b);

  core::SearchSpace search;
  search.fix("param_a", 5.0);

  auto bounds = search.get_effective_bounds(param_space);
  if (bounds.size() != 2) {
    std::cerr << "    expected 2 effective bounds\n";
    return false;
  }

  auto dim = search.get_optimization_dimension(param_space);
  if (dim != 1) {
    std::cerr << "    expected 1 optimization dimension, got " << dim << "\n";
    return false;
  }

  return true;
}

bool test_fixed_value_validation() {
  std::cout << "  fixed value validation\n";

  core::ParameterSpace param_space;

  core::ParameterDescriptor desc;
  desc.name = "param";
  desc.type = core::ParameterType::Continuous;
  desc.continuous_range = core::ContinuousRange{0.0, 1.0};
  param_space.add_descriptor(desc);

  // fixed value outside range should throw
  core::SearchSpace search;
  search.fix("param", 5.0);

  try {
    search.validate(param_space);
    std::cerr << "    fixed value outside range should throw\n";
    return false;
  } catch (const core::ParameterValidationError &) {
    // expected
  }

  return true;
}

bool test_integration_fixed_param() {
  std::cout << "  integration: fixed parameters\n";

  pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
  wrappers::problems::SphereProblem problem(5);

  auto search = std::make_shared<core::SearchSpace>();
  search->fix("population_size", std::int64_t{50});
  search->fix("variant", std::int64_t{2});
  search->fix("scaling_factor", 0.8);
  search->fix("crossover_rate", 0.9);

  pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
  optimizer.set_search_space(search);

  core::Budget budget;
  budget.generations = 3;

  auto result = optimizer.optimize(de_factory, problem, budget, 42UL);

  if (result.status != core::RunStatus::Success) {
    std::cerr << "    optimization failed: " << result.message << "\n";
    return false;
  }

  for (const auto &trial : result.trials) {
    auto pop_size =
        std::get<std::int64_t>(trial.parameters.at("population_size"));
    if (pop_size != 50) {
      std::cerr << "    population_size not fixed: " << pop_size << "\n";
      return false;
    }

    auto variant = std::get<std::int64_t>(trial.parameters.at("variant"));
    if (variant != 2) {
      std::cerr << "    variant not fixed: " << variant << "\n";
      return false;
    }

    auto sf = std::get<double>(trial.parameters.at("scaling_factor"));
    if (std::abs(sf - 0.8) > 1e-9) {
      std::cerr << "    scaling_factor not fixed: " << sf << "\n";
      return false;
    }

    auto cr = std::get<double>(trial.parameters.at("crossover_rate"));
    if (std::abs(cr - 0.9) > 1e-9) {
      std::cerr << "    crossover_rate not fixed: " << cr << "\n";
      return false;
    }
  }

  return true;
}

bool test_integration_custom_bounds() {
  std::cout << "  integration: custom bounds\n";

  pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
  wrappers::problems::SphereProblem problem(5);

  auto search = std::make_shared<core::SearchSpace>();
  search->optimize("scaling_factor", core::ContinuousRange{0.5, 0.6});
  search->optimize("crossover_rate", core::ContinuousRange{0.8, 0.9});

  pagmo_wrappers::PagmoCmaesHyperOptimizer optimizer;
  optimizer.set_search_space(search);

  core::Budget budget;
  budget.generations = 3;

  auto result = optimizer.optimize(de_factory, problem, budget, 42UL);

  if (result.status != core::RunStatus::Success) {
    std::cerr << "    optimization failed: " << result.message << "\n";
    return false;
  }

  for (const auto &trial : result.trials) {
    auto sf = std::get<double>(trial.parameters.at("scaling_factor"));
    if (sf < 0.5 || sf > 0.6) {
      std::cerr << "    scaling_factor out of custom bounds: " << sf << "\n";
      return false;
    }

    auto cr = std::get<double>(trial.parameters.at("crossover_rate"));
    if (cr < 0.8 || cr > 0.9) {
      std::cerr << "    crossover_rate out of custom bounds: " << cr << "\n";
      return false;
    }
  }

  return true;
}

} // namespace

int main() {
  int failures = 0;

  std::cout << "search space tests\n";

  if (!test_search_space_basic())
    failures++;
  if (!test_transforms())
    failures++;
  if (!test_validation())
    failures++;
  if (!test_transform_validation())
    failures++;
  if (!test_bounds_validation())
    failures++;
  if (!test_validate_and_clamp())
    failures++;
  if (!test_effective_bounds())
    failures++;
  if (!test_fixed_value_validation())
    failures++;
  if (!test_integration_fixed_param())
    failures++;
  if (!test_integration_custom_bounds())
    failures++;

  if (failures > 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }

  std::cout << "all search space tests passed\n";
  return 0;
}
