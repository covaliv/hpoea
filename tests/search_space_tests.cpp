#include "test_harness.hpp"

#include "hpoea/core/parameters.hpp"
#include "hpoea/core/search_space.hpp"

#include <cmath>
#include <string>

using hpoea::core::ParameterDescriptor;
using hpoea::core::ParameterSpace;
using hpoea::core::ParameterType;
using hpoea::core::SearchMode;

namespace {

ParameterSpace make_space() {
    ParameterSpace space;

    ParameterDescriptor cont;
    cont.name = "lr";
    cont.type = ParameterType::Continuous;
    cont.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
    space.add_descriptor(cont);

    ParameterDescriptor integer;
    integer.name = "pop";
    integer.type = ParameterType::Integer;
    integer.integer_range = hpoea::core::IntegerRange{1, 10};
    space.add_descriptor(integer);

    ParameterDescriptor categorical;
    categorical.name = "mode";
    categorical.type = ParameterType::Categorical;
    categorical.categorical_choices = {"a", "b"};
    space.add_descriptor(categorical);

    ParameterDescriptor boolean;
    boolean.name = "flag";
    boolean.type = ParameterType::Boolean;
    space.add_descriptor(boolean);

    return space;
}

}

int main() {
    hpoea::tests_v2::TestRunner runner;

    hpoea::core::SearchSpace space;
    space.fix("lr", 0.25);
    space.exclude("flag");
    space.optimize("pop", hpoea::core::IntegerRange{2, 6});
    space.optimize_choices("mode", {std::string{"a"}, std::string{"b"}});

    HPOEA_V2_CHECK(runner, !space.empty(), "search space stores configurations");
    HPOEA_V2_CHECK(runner, space.get("lr")->mode == SearchMode::fixed, "fixed mode stored");
    HPOEA_V2_CHECK(runner, space.get("flag")->mode == SearchMode::exclude, "exclude mode stored");
    HPOEA_V2_CHECK(runner, space.get("pop")->integer_bounds.has_value(), "integer bounds stored");
    HPOEA_V2_CHECK(runner, space.get("mode")->discrete_choices.size() == 2,
                   "discrete choices stored");


    {
        const double v = hpoea::core::inverse_transform(2.0, hpoea::core::Transform::log);
        HPOEA_V2_CHECK(runner, std::fabs(v - 100.0) < 1e-12, "log transform applies 10^x");
    }
    {
        const auto bounds = hpoea::core::transform_bounds({1.0, 8.0}, hpoea::core::Transform::log2);
        HPOEA_V2_CHECK(runner, std::fabs(bounds.lower - 0.0) < 1e-12,
                       "log2 bounds lower equals 0");
        HPOEA_V2_CHECK(runner, std::fabs(bounds.upper - 3.0) < 1e-12,
                       "log2 bounds upper equals 3");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace clamp_space;
        clamp_space.optimize("lr", hpoea::core::ContinuousRange{-1.0, 2.0});
        clamp_space.validate_and_clamp(params);
        const auto *cfg = clamp_space.get("lr");
        HPOEA_V2_CHECK(runner, cfg && cfg->continuous_bounds.has_value(),
                       "validate_and_clamp keeps config");
        HPOEA_V2_CHECK(runner, cfg->continuous_bounds->lower == 0.0 &&
                                  cfg->continuous_bounds->upper == 1.0,
                       "validate_and_clamp clamps to descriptor bounds");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace invalid;
        invalid.fix("missing", 1.0);
        bool threw = false;
        try {
            invalid.validate(params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate rejects unknown parameter");
    }

    {
        bool threw = false;
        try {
            hpoea::core::SearchSpace invalid;
            invalid.optimize("lr", hpoea::core::ContinuousRange{0.0, 1.0}, hpoea::core::Transform::log);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "log transform requires positive bounds");
    }

    {
        bool threw = false;
        try {
            hpoea::core::SearchSpace invalid;
            invalid.optimize_choices("mode", {});
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "optimize_choices rejects empty list");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace custom;
        custom.fix("lr", 0.5);
        const auto bounds = custom.get_effective_bounds(params);
        const auto dim = custom.get_optimization_dimension(params);
        HPOEA_V2_CHECK(runner, bounds.size() == params.descriptors().size(),
                       "effective bounds cover all descriptors");
        HPOEA_V2_CHECK(runner, dim == 3u, "optimization dimension excludes fixed parameter");
    }


    {
        const double v = hpoea::core::inverse_transform(3.0, hpoea::core::Transform::log2);
        HPOEA_V2_CHECK(runner, std::fabs(v - 8.0) < 1e-9, "log2 inverse_transform applies 2^x");
    }


    {
        const double v = hpoea::core::inverse_transform(3.0, hpoea::core::Transform::sqrt);
        HPOEA_V2_CHECK(runner, std::fabs(v - 9.0) < 1e-9, "sqrt inverse_transform applies x^2");
    }


    {
        bool threw = false;
        try {
            hpoea::core::SearchSpace invalid;
            invalid.optimize("p", hpoea::core::ContinuousRange{-1.0, 1.0}, hpoea::core::Transform::sqrt);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "sqrt transform rejects negative lower bound");
    }


    {
        bool threw = false;
        try {
            hpoea::core::SearchSpace valid;
            valid.optimize("p", hpoea::core::ContinuousRange{0.0, 1.0}, hpoea::core::Transform::sqrt);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, !threw, "sqrt transform accepts zero lower bound");
    }


    {
        bool threw = false;
        try {
            hpoea::core::SearchSpace invalid;
            invalid.optimize("p", hpoea::core::ContinuousRange{10.0, 1.0});
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "optimize rejects continuous lower > upper");
    }


    {
        bool threw = false;
        try {
            hpoea::core::SearchSpace invalid;
            invalid.optimize("p", hpoea::core::IntegerRange{10, 1});
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "optimize rejects integer lower > upper");
    }


    {
        ParameterSpace params;
        ParameterDescriptor desc;
        desc.name = "x";
        desc.type = ParameterType::Continuous;
        desc.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
        params.add_descriptor(desc);

        hpoea::core::SearchSpace search;
        search.fix("x", 5.0);

        bool threw = false;
        try {
            search.validate(params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate rejects fixed value outside parameter range");
    }




    {
        hpoea::core::SearchSpace space;
        hpoea::core::ParameterConfig config;
        config.mode = SearchMode::optimize;
        config.continuous_bounds = hpoea::core::ContinuousRange{0.1, 0.9};
        space.set("lr", config);
        HPOEA_V2_CHECK(runner, space.has("lr"), "set: config is stored");
        HPOEA_V2_CHECK(runner, space.get("lr")->mode == SearchMode::optimize, "set: mode is optimize");
        HPOEA_V2_CHECK(runner, space.get("lr")->continuous_bounds->lower == 0.1, "set: bounds stored correctly");
    }


    {
        hpoea::core::SearchSpace space;
        hpoea::core::ParameterConfig config;
        config.mode = SearchMode::optimize;
        config.integer_bounds = hpoea::core::IntegerRange{10, 1};
        bool threw = false;
        try {
            space.set("pop", config);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "set: rejects integer bounds lower > upper");
    }


    {
        hpoea::core::SearchSpace space;
        hpoea::core::ParameterConfig config;
        config.mode = SearchMode::optimize;
        config.continuous_bounds = hpoea::core::ContinuousRange{0.0, 1.0};
        config.transform = hpoea::core::Transform::log;
        bool threw = false;
        try {
            space.set("lr", config);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "set: rejects log transform with non-positive lower bound");
    }


    {
        hpoea::core::SearchSpace space;
        space.fix("lr", 0.5);
        HPOEA_V2_CHECK(runner, space.get("lr")->mode == SearchMode::fixed, "set overwrite: initially fixed");
        space.optimize("lr", hpoea::core::ContinuousRange{0.1, 0.9});
        HPOEA_V2_CHECK(runner, space.get("lr")->mode == SearchMode::optimize, "set overwrite: now optimize");
    }




    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace space;
        space.optimize_choices("pop", {std::string{"wrong_type"}});
        bool threw = false;
        try {
            space.validate(params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate: integer discrete_choices rejects wrong variant type");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace space;
        space.optimize_choices("pop", {std::int64_t{99}});
        bool threw = false;
        try {
            space.validate(params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate: integer discrete_choices rejects out-of-range value");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace space;
        space.optimize_choices("mode", {std::int64_t{1}});
        bool threw = false;
        try {
            space.validate(params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate: categorical discrete_choices rejects wrong variant type");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace space;
        space.optimize_choices("mode", {std::string{"c"}});
        bool threw = false;
        try {
            space.validate(params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate: categorical discrete_choices rejects invalid value");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace space;
        space.optimize("pop", hpoea::core::ContinuousRange{0.0, 1.0});
        bool threw = false;
        try {
            space.validate(params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate: continuous bounds on integer param throws");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace space;
        space.optimize("lr", hpoea::core::IntegerRange{0, 10});
        bool threw = false;
        try {
            space.validate(params);
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "validate: integer bounds on continuous param throws");
    }




    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace space;
        space.fix("lr", 0.5);
        space.exclude("flag");
        space.optimize("pop", hpoea::core::IntegerRange{2, 8});

        const auto bounds = space.get_effective_bounds(params);
        HPOEA_V2_CHECK(runner, bounds.size() == 4u, "effective_bounds: all 4 descriptors present");


        const hpoea::core::EffectiveBounds *lr_eb = nullptr;
        const hpoea::core::EffectiveBounds *pop_eb = nullptr;
        const hpoea::core::EffectiveBounds *flag_eb = nullptr;
        const hpoea::core::EffectiveBounds *mode_eb = nullptr;
        for (const auto &eb : bounds) {
            if (eb.name == "lr") lr_eb = &eb;
            else if (eb.name == "pop") pop_eb = &eb;
            else if (eb.name == "flag") flag_eb = &eb;
            else if (eb.name == "mode") mode_eb = &eb;
        }

        HPOEA_V2_CHECK(runner, lr_eb && lr_eb->mode == SearchMode::fixed,
                       "effective_bounds: lr is fixed mode");
        HPOEA_V2_CHECK(runner, flag_eb && flag_eb->mode == hpoea::core::SearchMode::exclude,
                       "effective_bounds: flag is exclude mode");
        HPOEA_V2_CHECK(runner, pop_eb && pop_eb->mode == SearchMode::optimize,
                       "effective_bounds: pop is optimize mode");
        HPOEA_V2_CHECK(runner, pop_eb && pop_eb->integer_bounds.has_value(),
                       "effective_bounds: pop has integer bounds");
        HPOEA_V2_CHECK(runner, pop_eb && pop_eb->integer_bounds->lower == 2 &&
                               pop_eb->integer_bounds->upper == 8,
                       "effective_bounds: pop bounds are [2, 8]");

        HPOEA_V2_CHECK(runner, mode_eb && mode_eb->mode == SearchMode::optimize,
                       "effective_bounds: unconfigured mode defaults to optimize");
    }


    {
        ParameterSpace params = make_space();
        hpoea::core::SearchSpace space;
        space.fix("lr", 0.5);
        space.exclude("flag");
        const auto dim = space.get_optimization_dimension(params);

        HPOEA_V2_CHECK(runner, dim == 2u, "optimization dimension: fixed + excluded reduce count");
    }


    {
        hpoea::core::SearchSpace space;
        bool threw = false;
        try {
            hpoea::core::ParameterConfig config;
            config.integer_bounds = hpoea::core::IntegerRange{10, 5};
            space.set("bad_int", std::move(config));
        } catch (const hpoea::core::ParameterValidationError &) {
            threw = true;
        }
        HPOEA_V2_CHECK(runner, threw, "set() rejects integer bounds with lower > upper");
    }


    {
        hpoea::core::ParameterSpace pspace;
        hpoea::core::ParameterDescriptor d;
        d.name = "x";
        d.type = hpoea::core::ParameterType::Integer;
        d.integer_range = hpoea::core::IntegerRange{1, 10};
        d.default_value = std::int64_t{5};
        pspace.add_descriptor(d);

        hpoea::core::SearchSpace space;
        space.optimize_choices("x", {std::int64_t{3}, std::int64_t{15}});
        bool threw = false;
        try { space.validate(pspace); }
        catch (const hpoea::core::ParameterValidationError &) { threw = true; }
        HPOEA_V2_CHECK(runner, threw, "validate rejects integer discrete choice outside range");
    }


    {
        hpoea::core::ParameterSpace pspace;
        hpoea::core::ParameterDescriptor d;
        d.name = "x";
        d.type = hpoea::core::ParameterType::Integer;
        d.integer_range = hpoea::core::IntegerRange{1, 10};
        d.default_value = std::int64_t{5};
        pspace.add_descriptor(d);

        hpoea::core::SearchSpace space;
        space.optimize_choices("x", {std::string("bad")});
        bool threw = false;
        try { space.validate(pspace); }
        catch (const hpoea::core::ParameterValidationError &) { threw = true; }
        HPOEA_V2_CHECK(runner, threw, "validate rejects string choice for integer parameter");
    }


    {
        hpoea::core::ParameterSpace pspace;
        hpoea::core::ParameterDescriptor d;
        d.name = "color";
        d.type = hpoea::core::ParameterType::Categorical;
        d.categorical_choices = {"red", "green", "blue"};
        d.default_value = std::string("red");
        pspace.add_descriptor(d);

        hpoea::core::SearchSpace space;
        space.optimize_choices("color", {std::string("red"), std::string("purple")});
        bool threw = false;
        try { space.validate(pspace); }
        catch (const hpoea::core::ParameterValidationError &) { threw = true; }
        HPOEA_V2_CHECK(runner, threw, "validate rejects invalid categorical choice");
    }


    {
        hpoea::core::ParameterSpace pspace;
        hpoea::core::ParameterDescriptor d;
        d.name = "color";
        d.type = hpoea::core::ParameterType::Categorical;
        d.categorical_choices = {"red", "green", "blue"};
        d.default_value = std::string("red");
        pspace.add_descriptor(d);

        hpoea::core::SearchSpace space;
        space.optimize_choices("color", {std::int64_t{1}});
        bool threw = false;
        try { space.validate(pspace); }
        catch (const hpoea::core::ParameterValidationError &) { threw = true; }
        HPOEA_V2_CHECK(runner, threw, "validate rejects integer choice for categorical parameter");
    }


    {
        hpoea::core::ParameterSpace pspace;
        hpoea::core::ParameterDescriptor d;
        d.name = "lr";
        d.type = hpoea::core::ParameterType::Continuous;
        d.continuous_range = hpoea::core::ContinuousRange{0.001, 1.0};
        d.default_value = 0.01;
        pspace.add_descriptor(d);

        d = {};
        d.name = "epochs";
        d.type = hpoea::core::ParameterType::Integer;
        d.integer_range = hpoea::core::IntegerRange{1, 100};
        d.default_value = std::int64_t{10};
        pspace.add_descriptor(d);


        hpoea::core::SearchSpace space;
        auto bounds = space.get_effective_bounds(pspace);
        HPOEA_V2_CHECK(runner, bounds.size() == 2, "get_effective_bounds returns 2 entries");


        bool found_lr = false;
        for (const auto &eb : bounds) {
            if (eb.name == "lr") {
                found_lr = true;
                HPOEA_V2_CHECK(runner, eb.mode == hpoea::core::SearchMode::optimize,
                               "get_effective_bounds: lr mode is optimize");
                HPOEA_V2_CHECK(runner, eb.continuous_bounds.has_value(),
                               "get_effective_bounds: lr has continuous_bounds");
                if (eb.continuous_bounds) {
                    HPOEA_V2_CHECK(runner, std::abs(eb.continuous_bounds->lower - 0.001) < 1e-10,
                                   "get_effective_bounds: lr lower = 0.001");
                    HPOEA_V2_CHECK(runner, std::abs(eb.continuous_bounds->upper - 1.0) < 1e-10,
                                   "get_effective_bounds: lr upper = 1.0");
                }
            }
        }
        HPOEA_V2_CHECK(runner, found_lr, "get_effective_bounds: found lr entry");
    }


    {
        hpoea::core::ParameterSpace pspace;
        hpoea::core::ParameterDescriptor d;
        d.name = "x";
        d.type = hpoea::core::ParameterType::Continuous;
        d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
        d.default_value = 0.5;
        pspace.add_descriptor(d);

        hpoea::core::SearchSpace space;
        space.fix("x", 0.5);
        auto bounds = space.get_effective_bounds(pspace);
        HPOEA_V2_CHECK(runner, bounds.size() == 1, "get_effective_bounds: fixed param still in list");
        HPOEA_V2_CHECK(runner, bounds[0].mode == hpoea::core::SearchMode::fixed,
                       "get_effective_bounds: fixed param has mode fixed");
        HPOEA_V2_CHECK(runner, !bounds[0].continuous_bounds.has_value(),
                       "get_effective_bounds: fixed param has no continuous_bounds");
    }


    {
        hpoea::core::ParameterSpace pspace;
        hpoea::core::ParameterDescriptor d;
        d.name = "a";
        d.type = hpoea::core::ParameterType::Continuous;
        d.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
        d.default_value = 0.5;
        pspace.add_descriptor(d);

        d = {};
        d.name = "b";
        d.type = hpoea::core::ParameterType::Integer;
        d.integer_range = hpoea::core::IntegerRange{1, 10};
        d.default_value = std::int64_t{5};
        pspace.add_descriptor(d);

        d = {};
        d.name = "c";
        d.type = hpoea::core::ParameterType::Boolean;
        d.default_value = true;
        pspace.add_descriptor(d);


        hpoea::core::SearchSpace empty_space;
        HPOEA_V2_CHECK(runner, empty_space.get_optimization_dimension(pspace) == 3,
                       "get_optimization_dimension: all params optimized = 3");


        hpoea::core::SearchSpace space;
        space.fix("a", 0.5);
        space.exclude("b");
        HPOEA_V2_CHECK(runner, space.get_optimization_dimension(pspace) == 1,
                       "get_optimization_dimension: 1 fixed + 1 excluded = 1 remaining");
    }

    return runner.summarize("search_space_tests");
}
