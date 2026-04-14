#pragma once

#include "hpoea/config/config_types.hpp"

#include <string>
#include <vector>

namespace hpoea::config {

// experimental config api for thesis/demo use
// names and fields may change before the library api is stabilized

enum class ExpansionDiagnosticSeverity {
    Error,
    Warning
};

struct ExpansionDiagnostic {
    ExpansionDiagnosticSeverity severity{ExpansionDiagnosticSeverity::Error};
    std::string path;
    std::string message;
};

struct ExpansionResult {
    // empty when expansion reports errors
    std::vector<ResolvedRunSpec> runs;
    std::vector<ExpansionDiagnostic> diagnostics;

    [[nodiscard]] bool has_errors() const noexcept;
    [[nodiscard]] bool ok() const noexcept;
};

[[nodiscard]] ExpansionResult expand_suite_config(const SuiteConfig &config);

} // namespace hpoea::config
