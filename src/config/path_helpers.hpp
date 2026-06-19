#pragma once

#include <cstddef>
#include <string>
#include <string_view>

// diagnostic-path helpers shared by parser, validator, and expander
// so paths stay identical
namespace hpoea::config::detail {

inline std::string join_path(std::string_view base,
                             std::string_view key) {
    if (base.empty()) {
        return std::string{key};
    }
    std::string path{base};
    path += '.';
    path += key;
    return path;
}

inline std::string join_index(std::string_view base,
                              std::size_t index) {
    std::string path{base};
    path += '[';
    path += std::to_string(index);
    path += ']';
    return path;
}

} // namespace hpoea::config::detail
