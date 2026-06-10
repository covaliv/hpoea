#include "test_harness.hpp"

#include "hpoea/core/parameters.hpp"
#include "hpoea/core/search_space.hpp"

#include <cmath>
#include <cstdint>
#include <string>

using hpoea::core::ContinuousRange;
using hpoea::core::IntegerRange;
using hpoea::core::ParameterValidationError;
using hpoea::core::Transform;

namespace {

bool nearly_equal(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol;
}

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    {
        struct ForwardRow {
            ContinuousRange in;
            Transform transform;
            double lower;
            double upper;
            const char *label;
        };
        const ForwardRow forward_rows[] = {
            {{1.0, 1000.0}, Transform::log,  0.0,  3.0,  "log [1,1000] -> [0,3]"},
            {{0.01, 100.0}, Transform::log,  -2.0, 2.0,  "log [0.01,100] -> [-2,2]"},
            {{1.0, 64.0},   Transform::log2, 0.0,  6.0,  "log2 [1,64] -> [0,6]"},
            {{2.0, 32.0},   Transform::log2, 1.0,  5.0,  "log2 [2,32] -> [1,5]"},
            {{0.0, 100.0},  Transform::sqrt, 0.0,  10.0, "sqrt [0,100] -> [0,10]"},
            {{4.0, 25.0},   Transform::sqrt, 2.0,  5.0,  "sqrt [4,25] -> [2,5]"},
            {{-5.0, 5.0},   Transform::none, -5.0, 5.0,  "none preserves bounds"},
        };
        for (const auto &r : forward_rows) {
            const auto b = hpoea::core::transform_bounds(r.in, r.transform);
            HPOEA_V2_CHECK(runner, nearly_equal(b.lower, r.lower), std::string(r.label) + " lower");
            HPOEA_V2_CHECK(runner, nearly_equal(b.upper, r.upper), std::string(r.label) + " upper");
        }
    }

    {
        HPOEA_V2_CHECK(runner, nearly_equal(hpoea::core::inverse_transform(-2.0, Transform::log), 0.01),
                       "log inverse transform applies 10^x");
        HPOEA_V2_CHECK(runner, nearly_equal(hpoea::core::inverse_transform(3.0, Transform::log2), 8.0),
                       "log2 inverse transform applies 2^x");
        HPOEA_V2_CHECK(runner, nearly_equal(hpoea::core::inverse_transform(3.0, Transform::sqrt), 9.0),
                       "sqrt inverse transform squares non-negative transformed value");
        HPOEA_V2_CHECK(runner, nearly_equal(hpoea::core::inverse_transform(-3.0, Transform::sqrt), 0.0),
                       "sqrt inverse transform clamps negative transformed value before squaring");
    }

    {
        struct IntClampRow {
            IntegerRange requested;
            IntegerRange descriptor;
            std::int64_t lower;
            std::int64_t upper;
            const char *label;
        };
        const IntClampRow int_rows[] = {
            {{5, 50},  {10, 100}, 10, 50, "clamp_bounds integer [5,50] vs [10,100] = [10,50]"},
            {{20, 80}, {10, 100}, 20, 80, "clamp_bounds integer [20,80] inside [10,100] unchanged"},
        };
        for (const auto &r : int_rows) {
            const auto result = hpoea::core::clamp_bounds(r.requested, r.descriptor);
            HPOEA_V2_CHECK(runner, result.lower == r.lower && result.upper == r.upper, r.label);
        }

        const auto cont = hpoea::core::clamp_bounds(ContinuousRange{-1.0, 5.0}, ContinuousRange{0.0, 3.0});
        HPOEA_V2_CHECK(runner, nearly_equal(cont.lower, 0.0) && nearly_equal(cont.upper, 3.0),
                       "clamp_bounds continuous [-1,5] vs [0,3] = [0,3]");
    }

    {
        struct ThrowRow { ContinuousRange in; Transform transform; const char *label; };
        const ThrowRow throw_rows[] = {
            {{0.0, 1.0},  Transform::log,  "log transform bounds reject non-positive lower bound"},
            {{-1.0, 1.0}, Transform::sqrt, "sqrt transform bounds reject negative lower bound"},
        };
        for (const auto &r : throw_rows) {
            bool threw = false;
            try {
                (void)hpoea::core::transform_bounds(r.in, r.transform);
            } catch (const ParameterValidationError &) {
                threw = true;
            }
            HPOEA_V2_CHECK(runner, threw, r.label);
        }
    }

    return runner.summarize("transform_bounds_tests");
}
