#include "hpoea/config/config_descriptors.hpp"

namespace hpoea::config {

const std::vector<ConfigTypeDescriptor> &algorithm_descriptors() {
    static const std::vector<ConfigTypeDescriptor> descriptors{
        {"de", true, true, {{"population_size", false}}},
        {"pso", true, true, {{"population_size", false}}},
        {"research_ea", false, false, {}}
    };
    return descriptors;
}

const std::vector<ConfigTypeDescriptor> &optimizer_descriptors() {
    static const std::vector<ConfigTypeDescriptor> descriptors{
        {"cmaes", true, false, {{"generations", false}}},
        {"pso", true, false, {{"generations", false}}},
        {"research_optimizer", false, false, {}}
    };
    return descriptors;
}

bool parameter_descriptors_complete(std::string_view type_id) {
    for (const auto &descriptor : algorithm_descriptors()) {
        if (descriptor.id == type_id) {
            return !descriptor.parameters.empty();
        }
    }
    for (const auto &descriptor : optimizer_descriptors()) {
        if (descriptor.id == type_id) {
            return !descriptor.parameters.empty();
        }
    }
    return false;
}

} // namespace hpoea::config
