#pragma once

#include <iostream>
#include <string>

namespace hpoea::tests_v2 {

struct TestRunner {
    int passed{0};
    int failed{0};

    void check(bool condition, const std::string &message) {
        if (condition) {
            ++passed;
        } else {
            ++failed;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    int summarize(const std::string &suite_name) const {
        std::cout << suite_name << ": " << passed << " passed, " << failed << " failed\n";
        return failed == 0 ? 0 : 1;
    }
};

}

#define HPOEA_V2_CHECK(runner, condition, message) \
    (runner).check((condition), std::string(message) + " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]")

#define HPOEA_V2_REQUIRE(runner, condition, message) \
    do { \
        if (!(condition)) { \
            (runner).check(false, std::string(message) + " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]"); \
            return; \
        } \
        (runner).check(true, std::string(message) + " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]"); \
    } while (0)
