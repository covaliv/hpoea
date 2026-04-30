#include "test_harness.hpp"

#include "hpoea/core/parameters.hpp"
#include "hpoea/core/search_space.hpp"

#include <cmath>
#include <string>

using hpoea::core::ContinuousRange;
using hpoea::core::IntegerRange;
using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;
using hpoea::core::ParameterValidationError;
using hpoea::core::SearchSpace;
using hpoea::core::Transform;

namespace {

bool nearly_equal(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol;
}

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    {
        const auto bounds = hpoea::core::transform_bounds({1.0, 1000.0}, Transform::log);
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.lower, 0.0),
                       "log transform bounds lower: log10(1) = 0");
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.upper, 3.0),
                       "log transform bounds upper: log10(1000) = 3");
    }

    {
        const auto bounds = hpoea::core::transform_bounds({1.0, 64.0}, Transform::log2);
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.lower, 0.0),
                       "log2 transform bounds lower: log2(1) = 0");
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.upper, 6.0),
                       "log2 transform bounds upper: log2(64) = 6");
    }

    {

        const auto bounds = hpoea::core::transform_bounds({0.0, 100.0}, Transform::sqrt);
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.lower, 0.0),
                       "sqrt transform bounds lower: sqrt(0) = 0");
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.upper, 10.0),
                       "sqrt transform bounds upper: sqrt(100) = 10");
    }

    {
        const auto bounds = hpoea::core::transform_bounds({-5.0, 5.0}, Transform::none);
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.lower, -5.0),
                       "none transform preserves lower bound");
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.upper, 5.0),
                       "none transform preserves upper bound");
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

        const auto bounds = hpoea::core::transform_bounds({0.01, 100.0}, Transform::log);
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.lower, -2.0),
                       "log transform bounds lower: log10(0.01) = -2");
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.upper, 2.0),
                       "log transform bounds upper: log10(100) = 2");
    }

    {

        const auto bounds = hpoea::core::transform_bounds({2.0, 32.0}, Transform::log2);
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.lower, 1.0),
                       "log2 transform bounds lower: log2(2) = 1");
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.upper, 5.0),
                       "log2 transform bounds upper: log2(32) = 5");
    }

    {

        const auto bounds = hpoea::core::transform_bounds({4.0, 25.0}, Transform::sqrt);
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.lower, 2.0),
                       "sqrt transform bounds lower: sqrt(4) = 2");
        HPOEA_V2_CHECK(runner, nearly_equal(bounds.upper, 5.0),
                       "sqrt transform bounds upper: sqrt(25) = 5");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "pop";
        desc.type = ParameterType::Integer;
        desc.integer_range = IntegerRange{10, 100};
        space.add_descriptor(desc);

        SearchSpace search;
        search.optimize("pop", IntegerRange{0, 50});
        search.validate_and_clamp(space);

        const auto *cfg = search.get("pop");
        HPOEA_V2_CHECK(runner, cfg != nullptr && cfg->integer_bounds.has_value(),
                       "integer clamp preserves config");
        HPOEA_V2_CHECK(runner, cfg->integer_bounds->lower == 10,
                       "integer clamp lower: max(0, 10) = 10");
        HPOEA_V2_CHECK(runner, cfg->integer_bounds->upper == 50,
                       "integer clamp upper: min(50, 100) = 50");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "pop";
        desc.type = ParameterType::Integer;
        desc.integer_range = IntegerRange{10, 100};
        space.add_descriptor(desc);

        SearchSpace search;
        search.optimize("pop", IntegerRange{0, 200});
        search.validate_and_clamp(space);

        const auto *cfg = search.get("pop");
        HPOEA_V2_CHECK(runner, cfg != nullptr && cfg->integer_bounds.has_value(),
                       "integer clamp wide range preserves config");
        HPOEA_V2_CHECK(runner, cfg->integer_bounds->lower == 10,
                       "integer clamp wide range lower: max(0, 10) = 10");
        HPOEA_V2_CHECK(runner, cfg->integer_bounds->upper == 100,
                       "integer clamp wide range upper: min(200, 100) = 100");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "pop";
        desc.type = ParameterType::Integer;
        desc.integer_range = IntegerRange{10, 100};
        space.add_descriptor(desc);

        SearchSpace search;
        search.optimize("pop", IntegerRange{0, 5});

        bool threw = false;
        try {
            search.validate_and_clamp(space);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw,
                       "integer clamp non-overlapping range throws ParameterValidationError");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "lr";
        desc.type = ParameterType::Continuous;
        desc.continuous_range = ContinuousRange{0.0, 1.0};
        space.add_descriptor(desc);

        SearchSpace search;
        search.optimize("lr", ContinuousRange{-0.5, 2.0});
        search.validate_and_clamp(space);

        const auto *cfg = search.get("lr");
        HPOEA_V2_CHECK(runner, cfg != nullptr && cfg->continuous_bounds.has_value(),
                       "continuous clamp preserves config");
        HPOEA_V2_CHECK(runner, nearly_equal(cfg->continuous_bounds->lower, 0.0),
                       "continuous clamp lower: max(-0.5, 0.0) = 0.0");
        HPOEA_V2_CHECK(runner, nearly_equal(cfg->continuous_bounds->upper, 1.0),
                       "continuous clamp upper: min(2.0, 1.0) = 1.0");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "lr";
        desc.type = ParameterType::Continuous;
        desc.continuous_range = ContinuousRange{0.0, 1.0};
        space.add_descriptor(desc);

        SearchSpace search;
        search.optimize("lr", ContinuousRange{2.0, 3.0});

        bool threw = false;
        try {
            search.validate_and_clamp(space);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw,
                       "continuous clamp non-overlapping range throws ParameterValidationError");
    }

    {
        auto result = hpoea::core::clamp_bounds(IntegerRange{5, 50}, IntegerRange{10, 100});
        HPOEA_V2_CHECK(runner, result.lower == 10 && result.upper == 50,
                       "clamp_bounds integer [5,50] vs [10,100] = [10,50]");
    }

    {
        auto result = hpoea::core::clamp_bounds(IntegerRange{20, 80}, IntegerRange{10, 100});
        HPOEA_V2_CHECK(runner, result.lower == 20 && result.upper == 80,
                       "clamp_bounds integer [20,80] inside [10,100] unchanged");
    }

    {
        auto result = hpoea::core::clamp_bounds(ContinuousRange{-1.0, 5.0}, ContinuousRange{0.0, 3.0});
        HPOEA_V2_CHECK(runner, nearly_equal(result.lower, 0.0) && nearly_equal(result.upper, 3.0),
                       "clamp_bounds continuous [-1,5] vs [0,3] = [0,3]");
    }

    {
        bool threw = false;
        try {
            (void)hpoea::core::transform_bounds({0.0, 1.0}, Transform::log);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw,
                       "log transform bounds reject non-positive lower bound");
    }

    {
        bool threw = false;
        try {
            (void)hpoea::core::transform_bounds({-1.0, 1.0}, Transform::sqrt);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw,
                       "sqrt transform bounds reject negative lower bound");
    }

    return runner.summarize("transform_bounds_tests");
}
