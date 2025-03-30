#pragma once

#include "hpoea/core/evolution_algorithm.hpp"
#include "hpoea/core/hyperparameter_optimizer.hpp"
#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace hpoea::core {

struct RunRecord {
    std::string experiment_id;
    std::string problem_id;
    AlgorithmIdentity evolutionary_algorithm;
    std::optional<AlgorithmIdentity> hyper_optimizer;
    ParameterSet algorithm_parameters;
    ParameterSet optimizer_parameters;
    RunStatus status{RunStatus::InternalError};
    double objective_value{0.0};
    BudgetUsage budget_usage{};
    unsigned long algorithm_seed{0};
    std::optional<unsigned long> optimizer_seed;
    std::string message;
};

class ILogger {
public:
    virtual ~ILogger() = default;

    virtual void log(const RunRecord &record) = 0;

    virtual void flush() = 0;
};

class JsonlLogger final : public ILogger {
public:
    explicit JsonlLogger(std::filesystem::path file_path);

    void log(const RunRecord &record) override;

    void flush() override;

    [[nodiscard]] const std::filesystem::path &path() const noexcept { return file_path_; }

private:
    std::filesystem::path file_path_;
    std::ofstream stream_;
};

[[nodiscard]] std::string serialize_run_record(const RunRecord &record);

} // namespace hpoea::core

