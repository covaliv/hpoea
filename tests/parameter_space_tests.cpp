#include "test_harness.hpp"
#include "test_utils.hpp"

#include "hpoea/core/parameters.hpp"

#include <functional>
#include <limits>
#include <string>
#include <vector>

using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSet;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;
using hpoea::core::ParameterValidationError;

namespace {

ParameterSpace make_space() {
    ParameterSpace space;

    ParameterDescriptor alpha;
    alpha.name = "alpha";
    alpha.type = ParameterType::Continuous;
    alpha.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    alpha.default_value = 0.5;
    space.add_descriptor(alpha);

    ParameterDescriptor beta;
    beta.name = "beta";
    beta.type = ParameterType::Integer;
    beta.integer_range = hpoea::core::IntegerRange{1, 10};
    beta.required = true;
    space.add_descriptor(beta);

    ParameterDescriptor mode;
    mode.name = "mode";
    mode.type = ParameterType::Categorical;
    mode.categorical_choices = {"fast", "slow"};
    mode.default_value = std::string{"fast"};
    space.add_descriptor(mode);

    ParameterDescriptor flag;
    flag.name = "flag";
    flag.type = ParameterType::Boolean;
    flag.default_value = true;
    space.add_descriptor(flag);

    return space;
}

}

int main() {
    hpoea::tests_v2::TestRunner runner;


    {
        // each add_descriptor rule that must reject a malformed descriptor
        struct AddCase {
            const char *label;
            std::function<void(ParameterSpace &)> add;
        };
        const std::vector<AddCase> add_cases = {
            {"empty descriptor name rejected", [](ParameterSpace &s) {
                 ParameterDescriptor d;
                 d.name = "";
                 d.type = ParameterType::Continuous;
                 d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
                 s.add_descriptor(d);
             }},
            {"duplicate descriptor name rejected", [](ParameterSpace &s) {
                 ParameterDescriptor d;
                 d.name = "dup";
                 d.type = ParameterType::Continuous;
                 d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
                 s.add_descriptor(d);
                 s.add_descriptor(d);
             }},
            {"continuous descriptor requires bounds", [](ParameterSpace &s) {
                 ParameterDescriptor d;
                 d.name = "alpha";
                 d.type = ParameterType::Continuous;
                 s.add_descriptor(d);
             }},
            {"integer descriptor lower>upper rejected", [](ParameterSpace &s) {
                 ParameterDescriptor d;
                 d.name = "beta";
                 d.type = ParameterType::Integer;
                 d.integer_range = hpoea::core::IntegerRange{2, 1};
                 s.add_descriptor(d);
             }},
            {"categorical descriptor requires choices", [](ParameterSpace &s) {
                 ParameterDescriptor d;
                 d.name = "mode";
                 d.type = ParameterType::Categorical;
                 s.add_descriptor(d);
             }},
            {"integer descriptor requires bounds", [](ParameterSpace &s) {
                 ParameterDescriptor d;
                 d.name = "beta";
                 d.type = ParameterType::Integer;
                 s.add_descriptor(d);
             }},
            {"continuous descriptor lower>upper rejected", [](ParameterSpace &s) {
                 ParameterDescriptor d;
                 d.name = "alpha";
                 d.type = ParameterType::Continuous;
                 d.continuous_range = hpoea::core::ContinuousRange{2.0, 1.0};
                 s.add_descriptor(d);
             }},
        };
        for (const auto &c : add_cases) {
            ParameterSpace space;
            bool threw = false;
            try {
                c.add(space);
            } catch (const ParameterValidationError &) {
                threw = true;
            }
            HPOEA_V2_CHECK(runner, threw, c.label);
        }
    }


    {
        ParameterSpace space = make_space();
        ParameterSet overrides;
        overrides.emplace("beta", std::int64_t{3});
        auto applied = space.apply_defaults(overrides);

        HPOEA_V2_CHECK(runner, applied.size() == 4, "apply_defaults populates all parameters");
        HPOEA_V2_CHECK(runner, std::get<double>(applied.at("alpha")) == 0.5,
                       "apply_defaults sets continuous default");
        HPOEA_V2_CHECK(runner, std::get<std::int64_t>(applied.at("beta")) == 3,
                       "apply_defaults preserves override");
        HPOEA_V2_CHECK(runner, std::get<std::string>(applied.at("mode")) == "fast",
                       "apply_defaults sets categorical default");
        HPOEA_V2_CHECK(runner, std::get<bool>(applied.at("flag")),
                       "apply_defaults sets boolean default");
    }


    {
        ParameterSpace space = make_space();
        ParameterSet values;
        values.emplace("alpha", std::int64_t{1});
        values.emplace("beta", std::int64_t{2});
        bool threw = false;
        try {
            space.validate(values);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, !threw, "validate accepts int64 literal for continuous descriptor");

        const auto applied = space.apply_defaults(values);
        HPOEA_V2_CHECK(runner, std::holds_alternative<double>(applied.at("alpha")),
                       "apply_defaults coerces int64 continuous value to double");
        HPOEA_V2_CHECK(runner, std::get<double>(applied.at("alpha")) == 1.0,
                       "coerced continuous value equals 1.0");
    }


    {
        // validate accepts a well-formed set
        // rejects each way it can be malformed
        struct ValidateCase {
            const char *label;
            bool expect_throw;
            std::function<void(ParameterSet &)> fill;
        };
        const std::vector<ValidateCase> validate_cases = {
            {"validate accepts correct types and bounds", false, [](ParameterSet &v) {
                 v.emplace("alpha", 0.25);
                 v.emplace("beta", std::int64_t{5});
                 v.emplace("mode", std::string{"slow"});
                 v.emplace("flag", false);
             }},
            {"validate rejects missing required parameter", true, [](ParameterSet &v) {
                 v.emplace("alpha", 0.25);
             }},
            {"validate rejects NaN continuous value", true, [](ParameterSet &v) {
                 v.emplace("beta", std::int64_t{2});
                 v.emplace("alpha", std::numeric_limits<double>::quiet_NaN());
             }},
            {"validate rejects invalid categorical choice", true, [](ParameterSet &v) {
                 v.emplace("beta", std::int64_t{2});
                 v.emplace("alpha", 0.5);
                 v.emplace("mode", std::string{"unknown"});
             }},
            {"validate rejects unknown parameter", true, [](ParameterSet &v) {
                 v.emplace("unknown", 1.0);
             }},
            {"validate rejects type mismatch", true, [](ParameterSet &v) {
                 v.emplace("beta", 1.5);
             }},
        };
        for (const auto &c : validate_cases) {
            ParameterSpace space = make_space();
            ParameterSet values;
            c.fill(values);
            bool threw = false;
            try {
                space.validate(values);
            } catch (const ParameterValidationError &) {
                threw = true;
            }
            HPOEA_V2_CHECK(runner, threw == c.expect_throw, c.label);
        }
    }


    {
        ParameterSpace space = make_space();
        ParameterSet bad;
        bad.emplace("alpha", 2.0);
        bad.emplace("beta", std::int64_t{2});
        bool threw = false;
        try {
            [[maybe_unused]] auto applied = space.apply_defaults(bad);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "apply_defaults rejects invalid override");
    }

    return runner.summarize("parameter_space_tests");
}
