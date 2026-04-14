#include "hpoea/config/config_parser.hpp"

#define TOML_EXCEPTIONS 0
#define TOML_ENABLE_FORMATTERS 0
#include <toml++/toml.hpp>

#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using hpoea::config::AlgorithmSpec;
using hpoea::config::BudgetConfig;
using hpoea::config::ConfigValue;
using hpoea::config::ExperimentSpec;
using hpoea::config::OptimizerSpec;
using hpoea::config::ParseDiagnostic;
using hpoea::config::ParseDiagnosticSeverity;
using hpoea::config::ParseResult;
using hpoea::config::ProblemSpec;
using hpoea::config::SearchChoiceList;
using hpoea::config::SearchParameterMode;
using hpoea::config::SearchParameterSpec;
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

std::string index_path(std::string_view base,
                       std::size_t index) {
    std::string path{base};
    path += '[';
    path += std::to_string(index);
    path += ']';
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

bool contains_key(std::initializer_list<std::string_view> allowed,
                  std::string_view key) {
    for (const auto allowed_key : allowed) {
        if (allowed_key == key) {
            return true;
        }
    }
    return false;
}

std::optional<SearchParameterMode> search_mode_from_string(std::string_view mode) {
    if (mode == "range") return SearchParameterMode::Range;
    if (mode == "integer_range") return SearchParameterMode::IntegerRange;
    if (mode == "choice") return SearchParameterMode::Choice;
    if (mode == "exclude") return SearchParameterMode::Exclude;
    return std::nullopt;
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

    void diagnose_unknown_keys(const toml::table &table,
                               std::string_view path,
                               std::initializer_list<std::string_view> allowed) {
        for (const auto &[raw_key, node] : table) {
            (void) node;
            const std::string key{raw_key.str()};
            if (!contains_key(allowed, key)) {
                error(join_path(path, key), "unknown field");
            }
        }
    }

    const toml::table *table_field(const toml::table &table,
                                   std::string_view key,
                                   std::string_view path,
                                   bool required) {
        const auto *node = table.get(key);
        if (!node) {
            if (required) {
                error(std::string{path}, "missing required table");
            }
            return nullptr;
        }
        const auto *value = node->as_table();
        if (!value) {
            error(std::string{path}, "expected table, got " + node_type_name(*node));
        }
        return value;
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

    std::optional<std::int64_t> read_integer(const toml::node &node,
                                             std::string_view path) {
        const auto value = node.value<std::int64_t>();
        if (!value) {
            error(std::string{path}, "expected integer, got " + node_type_name(node));
            return std::nullopt;
        }
        return *value;
    }

    template <typename IntegerType>
    std::optional<IntegerType> nonnegative_integer_field(const toml::table &table,
                                                         std::string_view key,
                                                         std::string_view path) {
        const auto *node = table.get(key);
        if (!node) {
            return std::nullopt;
        }
        const auto value = read_integer(*node, path);
        if (!value) {
            return std::nullopt;
        }
        if (*value < 0) {
            error(std::string{path}, "expected non-negative integer");
            return std::nullopt;
        }
        if (static_cast<unsigned long long>(*value)
            > static_cast<unsigned long long>(std::numeric_limits<IntegerType>::max())) {
            error(std::string{path}, "integer value is out of range");
            return std::nullopt;
        }
        return static_cast<IntegerType>(*value);
    }

    std::optional<double> read_double(const toml::node &node,
                                      std::string_view path) {
        if (const auto value = node.value<double>()) {
            return *value;
        }
        if (const auto value = node.value<std::int64_t>()) {
            return static_cast<double>(*value);
        }
        error(std::string{path}, "expected floating-point value, got " + node_type_name(node));
        return std::nullopt;
    }

    std::optional<ConfigValue> parse_config_array(const toml::array &array,
                                                  std::string_view path) {
        if (array.empty()) {
            error(std::string{path}, "arrays must not be empty");
            return std::nullopt;
        }
        const auto *first = array.get(0);
        if (!first || (first->type() != toml::node_type::integer
                       && first->type() != toml::node_type::floating_point)) {
            error(std::string{path}, "unsupported array element type: "
                                      + (first ? node_type_name(*first) : std::string{"unknown"}));
            return std::nullopt;
        }

        std::vector<std::int64_t> integer_values;
        std::vector<double> numeric_values;
        integer_values.reserve(array.size());
        numeric_values.reserve(array.size());
        bool all_integer = true;

        for (std::size_t i = 0; i < array.size(); ++i) {
            const auto *node = array.get(i);
            const auto element_path = index_path(path, i);
            if (!node) {
                error(element_path, "mixed-type arrays are not supported");
                return std::nullopt;
            }
            if (node->type() == toml::node_type::integer) {
                const auto value = *node->value<std::int64_t>();
                integer_values.push_back(value);
                numeric_values.push_back(static_cast<double>(value));
            } else if (node->type() == toml::node_type::floating_point) {
                all_integer = false;
                numeric_values.push_back(*node->value<double>());
            } else {
                error(element_path, "mixed-type arrays are not supported");
                return std::nullopt;
            }
        }

        if (all_integer) {
            return ConfigValue{std::move(integer_values)};
        }
        return ConfigValue{std::move(numeric_values)};
    }

    std::optional<ConfigValue> parse_config_value(const toml::node &node,
                                                  std::string_view path) {
        switch (node.type()) {
            case toml::node_type::integer: return ConfigValue{*node.value<std::int64_t>()};
            case toml::node_type::floating_point: return ConfigValue{*node.value<double>()};
            case toml::node_type::boolean: return ConfigValue{*node.value<bool>()};
            case toml::node_type::string: return ConfigValue{*node.value<std::string>()};
            case toml::node_type::array: return parse_config_array(*node.as_array(), path);
            default:
                error(std::string{path}, "unsupported value type: " + node_type_name(node));
                return std::nullopt;
        }
    }

    std::optional<hpoea::core::ParameterValue> parse_scalar_value(const toml::node &node,
                                                                  std::string_view path) {
        switch (node.type()) {
            case toml::node_type::integer: return hpoea::core::ParameterValue{*node.value<std::int64_t>()};
            case toml::node_type::floating_point: return hpoea::core::ParameterValue{*node.value<double>()};
            case toml::node_type::boolean: return hpoea::core::ParameterValue{*node.value<bool>()};
            case toml::node_type::string: return hpoea::core::ParameterValue{*node.value<std::string>()};
            default:
                error(std::string{path}, "expected scalar parameter value, got " + node_type_name(node));
                return std::nullopt;
        }
    }

    void parse_scalar_parameters(const toml::table &table,
                                 std::string_view path,
                                 hpoea::core::ParameterSet &parameters) {
        for (const auto &[raw_key, node] : table) {
            const std::string name{raw_key.str()};
            const auto value = parse_scalar_value(node, join_path(path, name));
            if (value) {
                parameters.emplace(name, *value);
            }
        }
    }

    std::optional<BudgetConfig> parse_budget(const toml::table &table,
                                             std::string_view path) {
        const auto before = error_count_;
        diagnose_unknown_keys(table, path, {"generations", "function_evaluations"});
        BudgetConfig budget;
        if (const auto value = nonnegative_integer_field<std::size_t>(table, "generations", join_path(path, "generations"))) {
            budget.generations = *value;
        }
        if (const auto value = nonnegative_integer_field<std::size_t>(table, "function_evaluations", join_path(path, "function_evaluations"))) {
            budget.function_evaluations = *value;
        }
        if (error_count_ != before) {
            return std::nullopt;
        }
        return budget;
    }

    std::optional<SearchChoiceList> parse_search_choices(const toml::array &array,
                                                         std::string_view path) {
        SearchChoiceList choices;
        choices.reserve(array.size());
        for (std::size_t i = 0; i < array.size(); ++i) {
            const auto *node = array.get(i);
            if (!node) {
                continue;
            }
            const auto value = parse_scalar_value(*node, index_path(path, i));
            if (!value) {
                return std::nullopt;
            }
            choices.push_back(*value);
        }
        return choices;
    }

    std::optional<SearchParameterSpec> parse_search_spec(const toml::table &table,
                                                         std::string_view path) {
        const auto before = error_count_;
        diagnose_unknown_keys(table, path, {"mode", "min", "max", "values"});
        const auto mode_name = string_field(table, "mode", join_path(path, "mode"), true);
        if (!mode_name) {
            return std::nullopt;
        }
        const auto mode = search_mode_from_string(*mode_name);
        if (!mode) {
            error(join_path(path, "mode"), "unsupported search mode: " + *mode_name);
            return std::nullopt;
        }

        SearchParameterSpec spec;
        spec.mode = *mode;
        const auto *min_node = table.get("min");
        const auto *max_node = table.get("max");
        spec.min_present = min_node != nullptr;
        spec.max_present = max_node != nullptr;
        if (spec.mode == SearchParameterMode::IntegerRange) {
            const auto min = min_node ? read_integer(*min_node, join_path(path, "min")) : std::nullopt;
            const auto max = max_node ? read_integer(*max_node, join_path(path, "max")) : std::nullopt;
            if (min && max) {
                spec.integer_range = hpoea::core::IntegerRange{*min, *max};
            }
        } else {
            const auto min = min_node ? read_double(*min_node, join_path(path, "min")) : std::nullopt;
            const auto max = max_node ? read_double(*max_node, join_path(path, "max")) : std::nullopt;
            if (min && max) {
                spec.continuous_range = hpoea::core::ContinuousRange{*min, *max};
            }
        }

        if (const auto *values_node = table.get("values")) {
            const auto values_path = join_path(path, "values");
            const auto *values = values_node->as_array();
            if (!values) {
                error(values_path, "expected array, got " + node_type_name(*values_node));
            } else if (const auto choices = parse_search_choices(*values, values_path)) {
                spec.choices = std::move(*choices);
            }
        }

        if (error_count_ != before) {
            return std::nullopt;
        }
        return spec;
    }

    void parse_root(const toml::table &root) {
        diagnose_unknown_keys(root, {}, {"schema_version", "suite", "problems", "algorithms",
                                         "optimizers", "experiments", "matrices"});
        config_ = SuiteConfig{};
        if (const auto version = nonnegative_integer_field<std::size_t>(root, "schema_version", "schema_version")) {
            config_.schema_version = *version;
        }
        if (root.get("matrices") != nullptr) {
            error("matrices", "unsupported section 'matrices'; see examples/configs/basic_experiment.toml");
        }
        parse_suite(root);
        parse_problems(root);
        parse_algorithms(root);
        parse_optimizers(root);
        parse_experiments(root);
    }

    void parse_suite(const toml::table &root) {
        const auto *table = table_field(root, "suite", "suite", true);
        if (!table) {
            return;
        }
        diagnose_unknown_keys(*table, "suite", {"name", "output_dir", "suite_seed", "repetitions"});
        if (const auto value = string_field(*table, "name", "suite.name", true)) {
            config_.name = *value;
        }
        if (const auto value = string_field(*table, "output_dir", "suite.output_dir", true)) {
            config_.output_dir = *value;
        }
        if (const auto value = nonnegative_integer_field<std::uint64_t>(*table, "suite_seed", "suite.suite_seed")) {
            config_.suite_seed = *value;
        }
        if (const auto value = nonnegative_integer_field<std::size_t>(*table, "repetitions", "suite.repetitions")) {
            config_.repetitions = *value;
        }
    }

    void parse_problems(const toml::table &root) {
        const auto *problems = table_field(root, "problems", "problems", false);
        if (!problems) {
            return;
        }
        for (const auto &[raw_key, node] : *problems) {
            const std::string id{raw_key.str()};
            const auto path = join_path("problems", id);
            const auto *table = node.as_table();
            if (!table) {
                error(path, "expected table, got " + node_type_name(node));
                continue;
            }
            ProblemSpec problem;
            problem.id = id;
            if (const auto type = string_field(*table, "type", join_path(path, "type"), true)) {
                problem.type = *type;
            }
            for (const auto &[param_key, param_node] : *table) {
                const std::string name{param_key.str()};
                if (name == "type") {
                    continue;
                }
                if (const auto value = parse_config_value(param_node, join_path(path, name))) {
                    problem.parameters.emplace(name, *value);
                }
            }
            config_.problems.push_back(std::move(problem));
        }
    }

    void parse_algorithms(const toml::table &root) {
        const auto *algorithms = table_field(root, "algorithms", "algorithms", false);
        if (!algorithms) {
            return;
        }
        for (const auto &[raw_key, node] : *algorithms) {
            const std::string id{raw_key.str()};
            const auto path = join_path("algorithms", id);
            const auto *table = node.as_table();
            if (!table) {
                error(path, "expected table, got " + node_type_name(node));
                continue;
            }
            diagnose_unknown_keys(*table, path, {"type", "fixed", "search"});
            AlgorithmSpec algorithm;
            algorithm.id = id;
            if (const auto type = string_field(*table, "type", join_path(path, "type"), true)) {
                algorithm.type = *type;
            }
            if (const auto *fixed = table_field(*table, "fixed", join_path(path, "fixed"), false)) {
                parse_scalar_parameters(*fixed, join_path(path, "fixed"), algorithm.fixed_parameters);
            }
            if (const auto *search = table_field(*table, "search", join_path(path, "search"), false)) {
                parse_search_parameters(*search, join_path(path, "search"), algorithm);
            }
            config_.algorithms.push_back(std::move(algorithm));
        }
    }

    void parse_search_parameters(const toml::table &table,
                                 std::string_view path,
                                 AlgorithmSpec &algorithm) {
        for (const auto &[raw_key, node] : table) {
            const std::string name{raw_key.str()};
            const auto field_path = join_path(path, name);
            const auto *spec_table = node.as_table();
            if (!spec_table) {
                error(field_path, "expected table, got " + node_type_name(node));
                continue;
            }
            if (const auto spec = parse_search_spec(*spec_table, field_path)) {
                algorithm.search_parameters.emplace(name, *spec);
            }
        }
    }

    void parse_optimizers(const toml::table &root) {
        const auto *optimizers = table_field(root, "optimizers", "optimizers", false);
        if (!optimizers) {
            return;
        }
        for (const auto &[raw_key, node] : *optimizers) {
            const std::string id{raw_key.str()};
            const auto path = join_path("optimizers", id);
            const auto *table = node.as_table();
            if (!table) {
                error(path, "expected table, got " + node_type_name(node));
                continue;
            }
            diagnose_unknown_keys(*table, path, {"type", "parameters"});
            OptimizerSpec optimizer;
            optimizer.id = id;
            if (const auto type = string_field(*table, "type", join_path(path, "type"), true)) {
                optimizer.type = *type;
            }
            if (const auto *params = table_field(*table, "parameters", join_path(path, "parameters"), false)) {
                parse_scalar_parameters(*params, join_path(path, "parameters"), optimizer.parameters);
            }
            config_.optimizers.push_back(std::move(optimizer));
        }
    }

    void parse_experiments(const toml::table &root) {
        const auto *node = root.get("experiments");
        if (!node) {
            return;
        }
        const auto *experiments = node->as_array();
        if (!experiments) {
            error("experiments", "expected array, got " + node_type_name(*node));
            return;
        }
        for (std::size_t i = 0; i < experiments->size(); ++i) {
            const auto *node = experiments->get(i);
            const auto path = index_path("experiments", i);
            const auto *table = node ? node->as_table() : nullptr;
            if (!table) {
                error(path, "expected table, got " + (node ? node_type_name(*node) : std::string{"unknown"}));
                continue;
            }
            parse_experiment(*table, path);
        }
    }

    void parse_experiment(const toml::table &table,
                          std::string_view path) {
        diagnose_unknown_keys(table, path, {"id", "problem", "algorithm", "optimizer", "repetitions",
                                            "seed", "output_name", "algorithm_budget", "optimizer_budget"});
        ExperimentSpec experiment;
        if (const auto value = string_field(table, "id", join_path(path, "id"), true)) {
            experiment.id = *value;
        }
        if (const auto value = string_field(table, "problem", join_path(path, "problem"), true)) {
            experiment.problem = *value;
        }
        if (const auto value = string_field(table, "algorithm", join_path(path, "algorithm"), true)) {
            experiment.algorithm = *value;
        }
        if (const auto value = string_field(table, "optimizer", join_path(path, "optimizer"), true)) {
            experiment.optimizer = *value;
        }
        if (const auto value = nonnegative_integer_field<std::size_t>(table, "repetitions", join_path(path, "repetitions"))) {
            experiment.repetitions = *value;
        }
        if (const auto value = nonnegative_integer_field<std::uint64_t>(table, "seed", join_path(path, "seed"))) {
            experiment.seed = *value;
        }
        if (const auto value = string_field(table, "output_name", join_path(path, "output_name"), false)) {
            experiment.output_name = *value;
        }
        if (const auto *budget = table_field(table, "algorithm_budget", join_path(path, "algorithm_budget"), false)) {
            experiment.algorithm_budget = parse_budget(*budget, join_path(path, "algorithm_budget"));
        }
        if (const auto *budget = table_field(table, "optimizer_budget", join_path(path, "optimizer_budget"), false)) {
            experiment.optimizer_budget = parse_budget(*budget, join_path(path, "optimizer_budget"));
        }
        config_.experiments.push_back(std::move(experiment));
    }

    std::string source_name_;
    SuiteConfig config_{};
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
    try {
        Parser parser{path.string()};
        return parser.parse_file(path);
    } catch (const std::exception &exception) {
        ParseResult result;
        result.diagnostics.push_back(ParseDiagnostic{ParseDiagnosticSeverity::Error,
                                                     path.string(),
                                                     {},
                                                     exception.what()});
        return result;
    }
}

ParseResult parse_config_string(std::string_view toml_text,
                                std::string source_name) {
    try {
        Parser parser{source_name};
        return parser.parse_string(toml_text);
    } catch (const std::exception &exception) {
        ParseResult result;
        result.diagnostics.push_back(ParseDiagnostic{ParseDiagnosticSeverity::Error,
                                                     std::move(source_name),
                                                     {},
                                                     exception.what()});
        return result;
    }
}

} // namespace hpoea::config
