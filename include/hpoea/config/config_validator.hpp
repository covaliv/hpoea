#pragma once

#include "hpoea/config/config_types.hpp"

#include <string>
#include <vector>

namespace hpoea::config {

// experimental config api for thesis/demo use
// names and fields may change before the library api is stabilized

enum class ValidationDiagnosticSeverity {
    Error,
    Warning
};

struct ValidationDiagnostic {
    ValidationDiagnosticSeverity severity{ValidationDiagnosticSeverity::Error};
    std::string path;
    std::string message;
};

struct ValidationResult {
    std::vector<ValidationDiagnostic> diagnostics;

    [[nodiscard]] bool has_errors() const noexcept;
    [[nodiscard]] bool ok() const noexcept;
};

[[nodiscard]] ValidationResult validate_suite_config(const SuiteConfig &config);

} // namespace hpoea::config
