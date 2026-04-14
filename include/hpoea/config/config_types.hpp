#pragma once

#include "hpoea/core/parameters.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace hpoea::config {

// experimental config api for thesis/demo use
// names and fields may change before the library api is stabilized

using ScalarConfigValue = core::ParameterValue;
using ConfigValue = std::variant<std::int64_t,
                                 double,
                                 bool,
                                 std::string,
                                 std::vector<std::int64_t>,
                                 std::vector<double>>;

using ProblemParameterSet = std::unordered_map<std::string, ConfigValue>;
using SearchChoiceList = std::vector<ScalarConfigValue>;

enum class SearchParameterMode {
    Range,
    IntegerRange,
    Choice,
    Exclude
};

struct SearchParameterSpec {
    SearchParameterMode mode{SearchParameterMode::Exclude};
    std::optional<core::ContinuousRange> continuous_range;
    std::optional<core::IntegerRange> integer_range;
    SearchChoiceList choices;
    bool min_present{false};
    bool max_present{false};
};

using SearchParameterSet = std::unordered_map<std::string, SearchParameterSpec>;

struct BudgetConfig {
    std::optional<std::size_t> generations;
    std::optional<std::size_t> function_evaluations;
};

struct ProblemSpec {
    std::string id;
    std::string type;
    ProblemParameterSet parameters;
};

struct AlgorithmSpec {
    std::string id;
    std::string type;
    core::ParameterSet fixed_parameters;
    SearchParameterSet search_parameters;
};

struct OptimizerSpec {
    std::string id;
    std::string type;
    core::ParameterSet parameters;
};

struct ExperimentSpec {
    std::string id;
    std::string problem;
    std::string algorithm;
    std::string optimizer;
    std::optional<std::size_t> repetitions;
    std::optional<std::uint64_t> seed;
    std::optional<std::string> output_name;
    std::optional<BudgetConfig> algorithm_budget;
    std::optional<BudgetConfig> optimizer_budget;
};

struct ResolvedRunSpec {
    std::string run_id;
    std::string experiment_id;
    std::string problem_id;
    std::string algorithm_id;
    std::string optimizer_id;
    std::size_t repetition_index{0};
    std::uint64_t seed{0};
    std::string output_name;
    std::filesystem::path planned_output_path;
    BudgetConfig algorithm_budget{};
    BudgetConfig optimizer_budget{};
};

struct SuiteConfig {
    std::size_t schema_version{1};
    std::string name;
    std::filesystem::path output_dir;
    std::optional<std::uint64_t> suite_seed;
    std::size_t repetitions{1};
    std::vector<ProblemSpec> problems;
    std::vector<AlgorithmSpec> algorithms;
    std::vector<OptimizerSpec> optimizers;
    std::vector<ExperimentSpec> experiments;
};

} // namespace hpoea::config
