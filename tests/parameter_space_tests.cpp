#include "test_harness.hpp"
#include "test_utils.hpp"

#include "hpoea/core/parameters.hpp"

#include <limits>
#include <string>

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
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "";
        desc.type = ParameterType::Continuous;
        desc.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
        bool threw = false;
        try {
            space.add_descriptor(desc);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "empty descriptor name rejected");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "dup";
        desc.type = ParameterType::Continuous;
        desc.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
        space.add_descriptor(desc);
        bool threw = false;
        try {
            space.add_descriptor(desc);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "duplicate descriptor name rejected");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "alpha";
        desc.type = ParameterType::Continuous;
        bool threw = false;
        try {
            space.add_descriptor(desc);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "continuous descriptor requires bounds");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "beta";
        desc.type = ParameterType::Integer;
        desc.integer_range = hpoea::core::IntegerRange{2, 1};
        bool threw = false;
        try {
            space.add_descriptor(desc);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "integer descriptor lower>upper rejected");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "mode";
        desc.type = ParameterType::Categorical;
        bool threw = false;
        try {
            space.add_descriptor(desc);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "categorical descriptor requires choices");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "beta";
        desc.type = ParameterType::Integer;
        bool threw = false;
        try {
            space.add_descriptor(desc);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "integer descriptor requires bounds");
    }

    {
        ParameterSpace space;
        ParameterDescriptor desc;
        desc.name = "alpha";
        desc.type = ParameterType::Continuous;
        desc.continuous_range = hpoea::core::ContinuousRange{2.0, 1.0};
        bool threw = false;
        try {
            space.add_descriptor(desc);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "continuous descriptor lower>upper rejected");
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
        values.emplace("alpha", 0.25);
        values.emplace("beta", std::int64_t{5});
        values.emplace("mode", std::string{"slow"});
        values.emplace("flag", false);
        bool threw = false;
        try {
            space.validate(values);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, !threw, "validate accepts correct types and bounds");
    }

    {
        ParameterSpace space = make_space();
        ParameterSet values;
        values.emplace("alpha", 0.25);
        bool threw = false;
        try {
            space.validate(values);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate rejects missing required parameter");
    }

    {
        ParameterSpace space = make_space();
        ParameterSet values;
        values.emplace("beta", std::int64_t{2});
        values.emplace("alpha", std::numeric_limits<double>::quiet_NaN());
        bool threw = false;
        try {
            space.validate(values);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate rejects NaN continuous value");
    }

    {
        ParameterSpace space = make_space();
        ParameterSet values;
        values.emplace("beta", std::int64_t{2});
        values.emplace("alpha", 0.5);
        values.emplace("mode", std::string{"unknown"});
        bool threw = false;
        try {
            space.validate(values);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate rejects invalid categorical choice");
    }

    {
        ParameterSpace space = make_space();
        ParameterSet values;
        values.emplace("unknown", 1.0);
        bool threw = false;
        try {
            space.validate(values);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate rejects unknown parameter");
    }

    {
        ParameterSpace space = make_space();
        ParameterSet values;
        values.emplace("beta", 1.5);
        bool threw = false;
        try {
            space.validate(values);
        } catch (const ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate rejects type mismatch");
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
