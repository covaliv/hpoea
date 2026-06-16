#pragma once

#include "hpoea/core/hyper_optimizer_base.hpp"

namespace hpoea::pagmo_wrappers {

// shared base for all pagmo-based hyper optimizers.
class PagmoHyperOptimizerBase : public core::HyperOptimizerBase {
protected:
    using core::HyperOptimizerBase::HyperOptimizerBase;
};

} // namespace hpoea::pagmo_wrappers
