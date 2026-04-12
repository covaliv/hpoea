#include "hpoea/config/config_parser.hpp"

#define TOML_EXCEPTIONS 0
#define TOML_ENABLE_FORMATTERS 0
#include <toml++/toml.hpp>

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::ExperimentSpec;
using hpoea::config::OptimizerSpec;
using hpoea::config::ParseDiagnostic;
using hpoea::config::ParseDiagnosticSeverity;
using hpoea::config::ParseResult;
using hpoea::config::ProblemSpec;
using hpoea::config::SuiteConfig;

std::string join_path(std::string_view base,
                      std::string_view key) {
    if (base.empty()) {
        return std::string{key};
    }
    std::string path{base};
    path += '.';
    path += key;
    return path;
}

std::string node_type_name(const toml::node &node) {
    switch (node.type()) {
        case toml::node_type::table: return "table";
        case toml::node_type::array: return "array";
        case toml::node_type::string: return "string";
        case toml::node_type::integer: return "integer";
        case toml::node_type::floating_point: return "floating_point";
        case toml::node_type::boolean: return "boolean";
        default: break;
    }
    return "unknown";
}

class Parser {
public:
    explicit Parser(std::string source_name)
        : source_name_(std::move(source_name)) {}

    ParseResult parse_string(std::string_view text) {
        return parse_document([&]() { return toml::parse(text, source_name_); });
    }

    ParseResult parse_file(const std::filesystem::path &path) {
        return parse_document([&]() { return toml::parse_file(path.string()); });
    }

private:
    template <typename ParseFn>
    ParseResult parse_document(ParseFn &&parse_fn) {
        const auto parsed = parse_fn();
        if (!parsed) {
            std::ostringstream stream;
            stream << parsed.error();
            error({}, stream.str());
            return finish();
        }
        parse_root(parsed.table());
        return finish();
    }

    ParseResult finish() {
        ParseResult result;
        result.diagnostics = std::move(diagnostics_);
        if (error_count_ == 0) {
            result.config = std::move(config_);
        }
        return result;
    }

    void error(std::string path,
               std::string message) {
        ++error_count_;
        diagnostics_.push_back(ParseDiagnostic{ParseDiagnosticSeverity::Error,
                                               source_name_,
                                               std::move(path),
                                               std::move(message)});
    }

    std::optional<std::string> string_field(const toml::table &table,
                                            std::string_view key,
                                            std::string_view path,
                                            bool required) {
        const auto *node = table.get(key);
        if (!node) {
            if (required) {
                error(std::string{path}, "missing required string value");
            }
            return std::nullopt;
        }
        const auto value = node->value<std::string>();
        if (!value) {
            error(std::string{path}, "expected string, got " + node_type_name(*node));
            return std::nullopt;
        }
        return *value;
    }

    std::optional<std::uint64_t> uint_field(const toml::table &table,
                                            std::string_view key,
                                            std::string_view path) {
        const auto *node = table.get(key);
        if (!node) {
            return std::nullopt;
        }
        const auto value = node->value<std::int64_t>();
        if (!value || *value < 0) {
            error(std::string{path}, "expected non-negative integer");
            return std::nullopt;
        }
        return static_cast<std::uint64_t>(*value);
    }

    void parse_root(const toml::table &root) {
        config_ = SuiteConfig{};
        if (const auto value = uint_field(root, "schema_version", "schema_version")) {
            config_.schema_version = static_cast<std::size_t>(*value);
        }
        parse_suite(root);
        parse_named_specs(root, "problems");
        parse_named_specs(root, "algorithms");
        parse_named_specs(root, "optimizers");
        parse_experiments(root);
    }

    void parse_suite(const toml::table &root) {
        const auto *node = root.get("suite");
        const auto *table = node ? node->as_table() : nullptr;
        if (!table) {
            error("suite", "missing required table");
            return;
        }
        if (const auto value = string_field(*table, "name", "suite.name", true)) {
            config_.name = *value;
        }
        if (const auto value = string_field(*table, "output_dir", "suite.output_dir", true)) {
            config_.output_dir = *value;
        }
        if (const auto value = uint_field(*table, "suite_seed", "suite.suite_seed")) {
            config_.suite_seed = *value;
        }
        if (const auto value = uint_field(*table, "repetitions", "suite.repetitions")) {
            config_.repetitions = static_cast<std::size_t>(*value);
        }
    }

    void parse_named_specs(const toml::table &root,
                           std::string_view section) {
        const auto *node = root.get(section);
        if (!node) {
            return;
        }
        const auto *table = node->as_table();
        if (!table) {
            error(std::string{section}, "expected table, got " + node_type_name(*node));
            return;
        }
        for (const auto &[raw_id, raw_spec] : *table) {
            const std::string id{raw_id.str()};
            const auto *spec_table = raw_spec.as_table();
            if (!spec_table) {
                error(join_path(section, id), "expected table, got " + node_type_name(raw_spec));
                continue;
            }
            const auto type = string_field(*spec_table, "type", join_path(join_path(section, id), "type"), true);
            if (!type) {
                continue;
            }
            if (section == "problems") {
                config_.problems.push_back(ProblemSpec{id, *type, {}});
            } else if (section == "algorithms") {
                config_.algorithms.push_back(AlgorithmSpec{id, *type, {}});
            } else {
                config_.optimizers.push_back(OptimizerSpec{id, *type, {}});
            }
        }
    }

    void parse_experiments(const toml::table &root) {
        const auto *node = root.get("experiments");
        const auto *array = node ? node->as_array() : nullptr;
        if (!array) {
            error("experiments", "expected array of tables");
            return;
        }
        for (std::size_t i = 0; i < array->size(); ++i) {
            const auto *item = array->get(i);
            const auto *table = item ? item->as_table() : nullptr;
            if (!table) {
                error("experiments", "expected table");
                continue;
            }
            ExperimentSpec experiment;
            experiment.id = string_field(*table, "id", "experiments.id", true).value_or({});
            experiment.problem = string_field(*table, "problem", "experiments.problem", true).value_or({});
            experiment.algorithm = string_field(*table, "algorithm", "experiments.algorithm", true).value_or({});
            experiment.optimizer = string_field(*table, "optimizer", "experiments.optimizer", true).value_or({});
            if (const auto value = uint_field(*table, "repetitions", "experiments.repetitions")) {
                experiment.repetitions = static_cast<std::size_t>(*value);
            }
            if (const auto value = uint_field(*table, "seed", "experiments.seed")) {
                experiment.seed = *value;
            }
            experiment.output_name = string_field(*table, "output_name", "experiments.output_name", false);
            config_.experiments.push_back(std::move(experiment));
        }
    }

    std::string source_name_;
    SuiteConfig config_;
    std::vector<ParseDiagnostic> diagnostics_;
    std::size_t error_count_{0};
};

} // namespace

namespace hpoea::config {

bool ParseResult::has_errors() const noexcept {
    for (const auto &diagnostic : diagnostics) {
        if (diagnostic.severity == ParseDiagnosticSeverity::Error) {
            return true;
        }
    }
    return false;
}

bool ParseResult::ok() const noexcept {
    return !has_errors();
}

ParseResult parse_config_file(const std::filesystem::path &path) {
    return Parser{path.string()}.parse_file(path);
}

ParseResult parse_config_string(std::string_view toml_text,
                                std::string source_name) {
    return Parser{std::move(source_name)}.parse_string(toml_text);
}

} // namespace hpoea::config
