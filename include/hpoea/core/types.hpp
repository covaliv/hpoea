#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>

namespace hpoea::core {

enum class RunStatus {
    Success,
    BudgetExceeded,
    FailedEvaluation,
    InvalidConfiguration,
    InternalError
};

struct Budget {
    std::optional<std::size_t> function_evaluations;
    std::optional<std::size_t> generations;
    std::optional<std::chrono::milliseconds> wall_time;
};

struct BudgetUsage {
    std::size_t function_evaluations{0};
    std::size_t generations{0};
    std::chrono::milliseconds wall_time{0};
};

struct AlgorithmIdentity {
    std::string family;
    std::string implementation;
    std::string version;
};

} // namespace hpoea::core

