#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace hpoea::core {

enum class ParameterType {
    Continuous,
    Integer,
    Boolean,
    Categorical
};

struct ContinuousRange {
    double lower{0.0};
    double upper{0.0};
};

struct IntegerRange {
    std::int64_t lower{0};
    std::int64_t upper{0};
};

using ParameterValue = std::variant<double, std::int64_t, bool, std::string>;
using ParameterSet = std::unordered_map<std::string, ParameterValue>;

struct ParameterDescriptor {
    std::string name;
    ParameterType type{ParameterType::Continuous};
    std::optional<ContinuousRange> continuous_range;
    std::optional<IntegerRange> integer_range;
    std::vector<std::string> categorical_choices;
    std::optional<ParameterValue> default_value;
    bool required{false};
};

class ParameterValidationError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ParameterSpace {
public:
    ParameterSpace() = default;

    void add_descriptor(ParameterDescriptor descriptor);

    [[nodiscard]] bool contains(const std::string &name) const noexcept;

    [[nodiscard]] const ParameterDescriptor &descriptor(const std::string &name) const;

    [[nodiscard]] const std::vector<ParameterDescriptor> &descriptors() const noexcept { return descriptors_; }

    [[nodiscard]] bool empty() const noexcept { return descriptors_.empty(); }

    [[nodiscard]] std::size_t size() const noexcept { return descriptors_.size(); }

    void validate(const ParameterSet &values) const;

    [[nodiscard]] ParameterSet apply_defaults(const ParameterSet &overrides) const;

private:
    void validate_value(const ParameterDescriptor &descriptor, const ParameterValue &value) const;

    std::vector<ParameterDescriptor> descriptors_;
    std::unordered_map<std::string, std::size_t> index_;
};

} // namespace hpoea::core

