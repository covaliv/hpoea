#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace hpoea::config {

struct ConfigParameterDescriptor {
    std::string name;
    bool required{false};
};

struct ConfigTypeDescriptor {
    std::string id;
    bool requires_pagmo{false};
    bool supports_search_space_tuning{false};
    std::vector<ConfigParameterDescriptor> parameters;
};

[[nodiscard]] const std::vector<ConfigTypeDescriptor> &algorithm_descriptors();
[[nodiscard]] const std::vector<ConfigTypeDescriptor> &optimizer_descriptors();
[[nodiscard]] bool parameter_descriptors_complete(std::string_view type_id);

} // namespace hpoea::config
