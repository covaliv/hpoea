#include "hpoea/wrappers/pagmo/hyper_optimizer_base.hpp"

namespace hpoea::pagmo_wrappers {

PagmoHyperOptimizerBase::PagmoHyperOptimizerBase(
    core::ParameterSpace space, core::AlgorithmIdentity identity)
    : parameter_space_(std::move(space)),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(std::move(identity)) {}

void PagmoHyperOptimizerBase::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

void PagmoHyperOptimizerBase::set_search_space(
    std::shared_ptr<core::SearchSpace> search_space) {
    search_space_ = std::move(search_space);
}

} // namespace hpoea::pagmo_wrappers
