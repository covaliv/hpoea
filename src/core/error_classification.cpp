#include "hpoea/core/error_classification.hpp"

namespace hpoea::core {

ClassifiedError classify_exception(const std::exception &ex) {
    if (dynamic_cast<const ParameterValidationError *>(&ex)) {
        return {RunStatus::InvalidConfiguration,
                ErrorInfo{"invalid_configuration", "parameter_validation", ex.what()}};
    }
    if (dynamic_cast<const std::invalid_argument *>(&ex)) {
        return {RunStatus::InvalidConfiguration,
                ErrorInfo{"invalid_configuration", "invalid_argument", ex.what()}};
    }
    if (dynamic_cast<const EvaluationFailure *>(&ex)) {
        return {RunStatus::FailedEvaluation,
                ErrorInfo{"evaluation_failure", "exception", ex.what()}};
    }
    return {RunStatus::InternalError,
            ErrorInfo{"internal_error", "exception", ex.what()}};
}

} // namespace hpoea::core
