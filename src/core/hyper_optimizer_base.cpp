#include "hpoea/core/hyper_optimizer_base.hpp"

#include <utility>

namespace hpoea::core {

HyperOptimizerBase::HyperOptimizerBase(ParameterSpace space, AlgorithmIdentity identity)
    : parameter_space_(std::move(space)),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(std::move(identity)) {}

HyperOptimizerBase::HyperOptimizerBase(const HyperOptimizerBase &other)
    : parameter_space_(other.parameter_space_),
      configured_parameters_(other.configured_parameters_),
      identity_(other.identity_),
      search_space_(other.search_space_ ? std::make_shared<SearchSpace>(*other.search_space_) : nullptr) {}

void HyperOptimizerBase::configure(const ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

void HyperOptimizerBase::set_search_space(std::shared_ptr<SearchSpace> search_space) {
    search_space_ = std::move(search_space);
}

} // namespace hpoea::core
