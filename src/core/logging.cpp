#include "hpoea/core/logging.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

std::string escape_json(std::string_view input) {
    std::string output;
    output.reserve(input.size() + input.size() / 2);
    for (const auto ch : input) {
        switch (ch) {
        case '"':
            output += "\\\"";
            break;
        case '\\':
            output += "\\\\";
            break;
        case '\b':
            output += "\\b";
            break;
        case '\f':
            output += "\\f";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                std::ostringstream oss;
                oss << "\\u" << std::hex << std::uppercase << std::setfill('0') << std::setw(4)
                    << static_cast<int>(static_cast<unsigned char>(ch));
                output += oss.str();
            } else {
                output += ch;
            }
            break;
        }
    }
    return output;
}

std::string run_status_to_string(hpoea::core::RunStatus status) {
    using hpoea::core::RunStatus;
    switch (status) {
    case RunStatus::Success:
        return "success";
    case RunStatus::BudgetExceeded:
        return "budget_exceeded";
    case RunStatus::FailedEvaluation:
        return "failed_evaluation";
    case RunStatus::InvalidConfiguration:
        return "invalid_configuration";
    case RunStatus::InternalError:
        return "internal_error";
    default:
        return "unknown";
    }
}

std::string serialize_double(double value) {
    if (std::isnan(value)) return "null";
    if (std::isinf(value)) return value > 0 ? "1e308" : "-1e308";
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << std::setprecision(17) << value;
    return oss.str();
}

std::string serialize_algorithm_identity(const hpoea::core::AlgorithmIdentity &identity) {
    std::ostringstream oss;
    oss << '{'
        << "\"family\":\"" << escape_json(identity.family) << "\",";
    oss << "\"implementation\":\"" << escape_json(identity.implementation) << "\",";
    oss << "\"version\":\"" << escape_json(identity.version) << "\"";
    oss << '}';
    return oss.str();
}

std::string serialize_parameter_value(const hpoea::core::ParameterValue &value) {
    return std::visit(
        [](const auto &val) -> std::string {
            using ValueType = std::decay_t<decltype(val)>;
            if constexpr (std::is_same_v<ValueType, double>) {
                return serialize_double(val);
            } else if constexpr (std::is_same_v<ValueType, std::int64_t>) {
                return std::to_string(val);
            } else if constexpr (std::is_same_v<ValueType, bool>) {
                return val ? "true" : "false";
            } else if constexpr (std::is_same_v<ValueType, std::string>) {
                return "\"" + escape_json(val) + "\"";
            } else {
                static_assert(sizeof(ValueType) == 0, "Unhandled ParameterValue variant type");
            }
        },
        value);
}

std::string serialize_parameter_set(const hpoea::core::ParameterSet &parameters) {
    if (parameters.empty()) {
        return "{}";
    }

    std::vector<std::pair<std::string, hpoea::core::ParameterValue>> ordered(parameters.begin(), parameters.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
    });

    std::ostringstream oss;
    oss << '{';
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        const auto &[name, value] = ordered[index];
        oss << "\"" << escape_json(name) << "\":" << serialize_parameter_value(value);
        if (index + 1 < ordered.size()) {
            oss << ',';
        }
    }
    oss << '}';
    return oss.str();
}

} // namespace

namespace hpoea::core {

JsonlLogger::JsonlLogger(std::filesystem::path file_path, bool auto_flush)
    : file_path_(std::move(file_path)), auto_flush_(auto_flush) {
    stream_.open(file_path_, std::ios::out | std::ios::app);
    if (!stream_.is_open()) {
        throw std::runtime_error("failed to open log file: " + file_path_.string());
    }
}

void JsonlLogger::log(const RunRecord &record) {
    if (!stream_.is_open()) {
        stream_.clear();
        stream_.open(file_path_, std::ios::out | std::ios::app);
        if (!stream_.is_open()) {
            throw std::runtime_error("failed to reopen log file: " + file_path_.string());
        }
    }

    stream_ << serialize_run_record(record) << '\n';
    if (!stream_.good()) {
        throw std::runtime_error("failed to write log record to: " + file_path_.string());
    }

    ++records_written_;

    if (auto_flush_) {
        stream_.flush();
    }
}

void JsonlLogger::flush() {
    if (stream_.is_open()) {
        stream_.flush();
    }
}

std::string serialize_run_record(const RunRecord &record) {
    std::ostringstream oss;
    oss << '{';
    oss << "\"experiment_id\":\"" << escape_json(record.experiment_id) << "\",";
    oss << "\"problem_id\":\"" << escape_json(record.problem_id) << "\",";
    oss << "\"evolutionary_algorithm\":" << serialize_algorithm_identity(record.evolutionary_algorithm) << ',';
    if (record.hyper_optimizer.has_value()) {
        oss << "\"hyper_optimizer\":" << serialize_algorithm_identity(*record.hyper_optimizer) << ',';
    } else {
        oss << "\"hyper_optimizer\":null,";
    }
    oss << "\"algorithm_parameters\":" << serialize_parameter_set(record.algorithm_parameters) << ',';
    oss << "\"optimizer_parameters\":" << serialize_parameter_set(record.optimizer_parameters) << ',';
    oss << "\"status\":\"" << escape_json(run_status_to_string(record.status)) << "\",";
    oss << "\"objective_value\":" << serialize_double(record.objective_value) << ',';
    oss << "\"budget_usage\":{"
        << "\"function_evaluations\":" << record.budget_usage.function_evaluations << ','
        << "\"generations\":" << record.budget_usage.generations << ','
        << "\"wall_time_ms\":" << record.budget_usage.wall_time.count() << "},";
    oss << "\"algorithm_seed\":" << record.algorithm_seed << ',';
    if (record.optimizer_seed.has_value()) {
        oss << "\"optimizer_seed\":" << *record.optimizer_seed << ',';
    } else {
        oss << "\"optimizer_seed\":null,";
    }
    oss << "\"message\":\"" << escape_json(record.message) << "\"";
    oss << '}';
    return oss.str();
}

} // namespace hpoea::core

