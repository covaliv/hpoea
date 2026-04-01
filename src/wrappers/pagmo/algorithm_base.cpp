#include "hpoea/wrappers/pagmo/algorithm_base.hpp"

#include <utility>

namespace hpoea::pagmo_wrappers {

PagmoAlgorithmBase::PagmoAlgorithmBase(core::ParameterSpace space,
                                       core::AlgorithmIdentity identity)
    : parameter_space_(std::move(space)),
      configured_parameters_(parameter_space_.apply_defaults({})),
      identity_(std::move(identity)) {}

void PagmoAlgorithmBase::configure(const core::ParameterSet &parameters) {
    configured_parameters_ = parameter_space_.apply_defaults(parameters);
    parameter_space_.validate(configured_parameters_);
}

PagmoAlgorithmFactoryBase::PagmoAlgorithmFactoryBase(core::ParameterSpace space,
                                                     core::AlgorithmIdentity identity)
    : parameter_space_(std::move(space)),
      identity_(std::move(identity)) {}

} // namespace hpoea::pagmo_wrappers
