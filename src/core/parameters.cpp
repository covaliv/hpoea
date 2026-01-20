#include "hpoea/core/parameters.hpp"

#include <algorithm>
#include <sstream>

namespace {

std::string parameter_type_to_string(hpoea::core::ParameterType type) {
    using hpoea::core::ParameterType;
    switch (type) {
    case ParameterType::Continuous:
        return "continuous";
    case ParameterType::Integer:
        return "integer";
    case ParameterType::Boolean:
        return "boolean";
    case ParameterType::Categorical:
        return "categorical";
    }
    return "unknown";
}

} // namespace

namespace hpoea::core {

void ParameterSpace::add_descriptor(ParameterDescriptor descriptor) {
    if (descriptor.name.empty()) {
        throw ParameterValidationError("Parameter descriptor name must not be empty");
    }
    if (contains(descriptor.name)) {
        throw ParameterValidationError("Parameter descriptor already exists: " + descriptor.name);
    }

    if (descriptor.type == ParameterType::Continuous && descriptor.continuous_range.has_value()) {
        const auto &range = *descriptor.continuous_range;
        if (range.lower > range.upper) {
            throw ParameterValidationError(
                "Continuous parameter '" + descriptor.name + "' has lower bound > upper bound");
        }
    }

    if (descriptor.type == ParameterType::Integer && descriptor.integer_range.has_value()) {
        const auto &range = *descriptor.integer_range;
        if (range.lower > range.upper) {
            throw ParameterValidationError(
                "Integer parameter '" + descriptor.name + "' has lower bound > upper bound");
        }
    }

    if (descriptor.type == ParameterType::Categorical && descriptor.categorical_choices.empty()) {
        throw ParameterValidationError(
            "Categorical parameter '" + descriptor.name + "' requires at least one choice");
    }

    index_[descriptor.name] = descriptors_.size();
    descriptors_.push_back(std::move(descriptor));
}

bool ParameterSpace::contains(const std::string &name) const noexcept {
    return index_.find(name) != index_.end();
}

const ParameterDescriptor &ParameterSpace::descriptor(const std::string &name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw ParameterValidationError("Unknown parameter: " + name);
    }
    return descriptors_[it->second];
}

void ParameterSpace::validate(const ParameterSet &values) const {
    for (const auto &[name, value] : values) {
        const auto &desc = descriptor(name);
        validate_value(desc, value);
    }

    for (const auto &desc : descriptors_) {
        if (desc.required && !values.contains(desc.name)) {
            throw ParameterValidationError("Missing required parameter: " + desc.name);
        }
    }
}

ParameterSet ParameterSpace::apply_defaults(const ParameterSet &overrides) const {
    ParameterSet result;

    for (const auto &[name, value] : overrides) {
        const auto &desc = descriptor(name);
        validate_value(desc, value);
        result.emplace(name, value);
    }

    for (const auto &desc : descriptors_) {
        if (result.contains(desc.name)) {
            continue;
        }

        if (desc.default_value.has_value()) {
            validate_value(desc, *desc.default_value);
            result.emplace(desc.name, *desc.default_value);
        } else if (desc.required) {
            throw ParameterValidationError("Missing required parameter: " + desc.name);
        }
    }

    return result;
}

void ParameterSpace::validate_value(const ParameterDescriptor &descriptor, const ParameterValue &value) const {
    std::stringstream message;
    message << "Parameter '" << descriptor.name << "' expects type " << parameter_type_to_string(descriptor.type);

    switch (descriptor.type) {
    case ParameterType::Continuous: {
        if (!std::holds_alternative<double>(value)) {
            message << " but received mismatched variant type";
            throw ParameterValidationError(message.str());
        }
        const auto numeric = std::get<double>(value);
        if (descriptor.continuous_range.has_value()) {
            const auto &range = *descriptor.continuous_range;
            if (numeric < range.lower || numeric > range.upper) {
                message << " outside bounds [" << range.lower << ", " << range.upper << "]";
                throw ParameterValidationError(message.str());
            }
        }
        break;
    }
    case ParameterType::Integer: {
        if (!std::holds_alternative<std::int64_t>(value)) {
            message << " but received mismatched variant type";
            throw ParameterValidationError(message.str());
        }
        const auto numeric = std::get<std::int64_t>(value);
        if (descriptor.integer_range.has_value()) {
            const auto &range = *descriptor.integer_range;
            if (numeric < range.lower || numeric > range.upper) {
                message << " outside bounds [" << range.lower << ", " << range.upper << "]";
                throw ParameterValidationError(message.str());
            }
        }
        break;
    }
    case ParameterType::Boolean: {
        if (!std::holds_alternative<bool>(value)) {
            message << " but received mismatched variant type";
            throw ParameterValidationError(message.str());
        }
        break;
    }
    case ParameterType::Categorical: {
        if (!std::holds_alternative<std::string>(value)) {
            message << " but received mismatched variant type";
            throw ParameterValidationError(message.str());
        }
        const auto &label = std::get<std::string>(value);
        const auto &choices = descriptor.categorical_choices;
        const auto iter = std::find(choices.begin(), choices.end(), label);
        if (iter == choices.end()) {
            message << " with invalid choice '" << label << "'";
            throw ParameterValidationError(message.str());
        }
        break;
    }
    }
}

} // namespace hpoea::core

