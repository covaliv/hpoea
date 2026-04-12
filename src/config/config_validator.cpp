#include "hpoea/config/config_validator.hpp"

namespace hpoea::config {

bool ValidationResult::has_errors() const noexcept {
    for (const auto &diagnostic : diagnostics) {
        if (diagnostic.severity == ValidationDiagnosticSeverity::Error) {
            return true;
        }
    }
    return false;
}

bool ValidationResult::ok() const noexcept {
    return !has_errors();
}

ValidationResult validate_suite_config(const SuiteConfig &config) {
    ValidationResult result;
    if (config.name.empty()) {
        result.diagnostics.push_back({ValidationDiagnosticSeverity::Error, "suite.name", "suite name must not be empty"});
    }
    if (config.output_dir.empty()) {
        result.diagnostics.push_back({ValidationDiagnosticSeverity::Error, "suite.output_dir", "suite output_dir must not be empty"});
    }
    if (config.experiments.empty()) {
        result.diagnostics.push_back({ValidationDiagnosticSeverity::Error, "experiments", "at least one explicit experiment is required"});
    }
    return result;
}

} // namespace hpoea::config
