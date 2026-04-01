#pragma once

#include "hpoea/core/parameters.hpp"
#include "hpoea/core/types.hpp"

#include <stdexcept>

namespace hpoea::core {

class EvaluationFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ClassifiedError {
    RunStatus status;
    ErrorInfo error_info;
};

ClassifiedError classify_exception(const std::exception &ex);

} // namespace hpoea::core
