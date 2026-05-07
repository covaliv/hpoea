#pragma once

#include "hpoea/config/config_types.hpp"

#include <string>

namespace hpoea::cli {

struct ComponentDispatch {
    std::string id;
    std::string type;
    std::string backend;
    std::string dispatch;
    bool runnable{false};
};

struct RunDispatch {
    ComponentDispatch problem;
    ComponentDispatch algorithm;
    ComponentDispatch optimizer;
    bool runnable{false};
};

[[nodiscard]] RunDispatch annotate_run_dispatch(const config::SuiteConfig &config,
                                                const config::ResolvedRunSpec &run);

} // namespace hpoea::cli
