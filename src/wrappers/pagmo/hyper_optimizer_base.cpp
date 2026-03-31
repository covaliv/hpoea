#include "hpoea/wrappers/pagmo/hyper_optimizer_base.hpp"

namespace hpoea::pagmo_wrappers {

PagmoHyperOptimizerBase::PagmoHyperOptimizerBase(
    core::ParameterSpace space, core::AlgorithmIdentity identity)
    : parameter_space_(std::move(space)),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(std::move(identity)) {}

PagmoHyperOptimizerBase::PagmoHyperOptimizerBase(const PagmoHyperOptimizerBase &other)
    : parameter_space_(other.parameter_space_),
      configured_parameters_(other.configured_parameters_),
      identity_(other.identity_),
      search_space_(other.search_space_
                       ? std::make_shared<core::SearchSpace>(*other.search_space_)
                       : nullptr) {}

void PagmoHyperOptimizerBase::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

void PagmoHyperOptimizerBase::set_search_space(
    std::shared_ptr<core::SearchSpace> search_space) {
    search_space_ = std::move(search_space);
}

} // namespace hpoea::pagmo_wrappers
