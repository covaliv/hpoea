#include "test_harness.hpp"

#include "hpoea/config/config_descriptors.hpp"

int main() {
    hpoea::tests_v2::TestRunner runner;

    HPOEA_V2_CHECK(runner, !hpoea::config::algorithm_descriptors().empty(),
                   "algorithm descriptors are registered");
    HPOEA_V2_CHECK(runner, hpoea::config::parameter_descriptors_complete("de"),
                   "known descriptor reports parameters");
    HPOEA_V2_CHECK(runner, !hpoea::config::parameter_descriptors_complete("unknown"),
                   "unknown descriptor is incomplete");

    return runner.summarize("config_descriptor_tests");
}
