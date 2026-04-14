#pragma once

#include "hpoea/config/config_types.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hpoea::config {

// experimental config api for thesis/demo use
// names and fields may change before the library api is stabilized

enum class ParseDiagnosticSeverity {
    Error,
    Warning
};

struct ParseDiagnostic {
    ParseDiagnosticSeverity severity{ParseDiagnosticSeverity::Error};
    std::string source;
    std::string path;
    std::string message;
};

struct ParseResult {
    std::optional<SuiteConfig> config;
    std::vector<ParseDiagnostic> diagnostics;

    [[nodiscard]] bool has_errors() const noexcept;
    [[nodiscard]] bool ok() const noexcept;
};

[[nodiscard]] ParseResult parse_config_file(const std::filesystem::path &path);
[[nodiscard]] ParseResult parse_config_string(std::string_view toml_text,
                                              std::string source_name = "<string>");

} // namespace hpoea::config
