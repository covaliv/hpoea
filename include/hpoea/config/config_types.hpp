#pragma once

#include "hpoea/core/parameters.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace hpoea::config {

using ProblemParameterSet = core::ParameterSet;
using OptimizerParameterSet = core::ParameterSet;

struct ProblemSpec {
    std::string id;
    std::string type;
    ProblemParameterSet parameters;
};

struct AlgorithmSpec {
    std::string id;
    std::string type;
    core::ParameterSet fixed_parameters;
};

struct OptimizerSpec {
    std::string id;
    std::string type;
    OptimizerParameterSet parameters;
};

struct ExperimentSpec {
    std::string id;
    std::string problem;
    std::string algorithm;
    std::string optimizer;
    std::optional<std::size_t> repetitions;
    std::optional<std::uint64_t> seed;
    std::optional<std::string> output_name;
};

struct BudgetConfig {
    std::optional<std::size_t> generations;
    std::optional<std::size_t> function_evaluations;
};

struct ResolvedRunSpec {
    std::string run_id;
    std::string experiment_id;
    std::size_t repetition_index{0};
    std::uint64_t seed{0};
    std::filesystem::path planned_output_path;
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
