#include "test_harness.hpp"

#include "hpoea/core/parameters.hpp"
#include "hpoea/core/search_space.hpp"

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

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
        // every validation/construction rule that must fail loud or succeed
        // one row per rule
        // the fn carries the full setup so shapes can differ
        const auto params = make_space();
        struct Case {
            const char *label;
            bool expect_throw;
            std::function<void()> fn;
        };
        const std::vector<Case> cases = {
            {"validate rejects optimize continuous bounds outside descriptor range", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("lr", hpoea::core::ContinuousRange{-1.0, 2.0});
                s.validate(params);
            }},
            {"validate rejects optimize integer bounds outside descriptor range", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("pop", hpoea::core::IntegerRange{0, 50});
                s.validate(params);
            }},
            {"validate accepts optimize bounds contained in descriptor range", false, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("lr", hpoea::core::ContinuousRange{0.1, 0.9});
                s.optimize("pop", hpoea::core::IntegerRange{2, 8});
                s.validate(params);
            }},
            {"validate rejects unknown parameter", true, [&] {
                hpoea::core::SearchSpace s;
                s.fix("missing", 1.0);
                s.validate(params);
            }},
            {"log transform requires positive bounds", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("lr", hpoea::core::ContinuousRange{0.0, 1.0}, hpoea::core::Transform::log);
            }},
            {"optimize_choices rejects empty list", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize_choices("mode", {});
            }},
            {"sqrt transform rejects negative lower bound", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("p", hpoea::core::ContinuousRange{-1.0, 1.0}, hpoea::core::Transform::sqrt);
            }},
            {"sqrt transform accepts zero lower bound", false, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("p", hpoea::core::ContinuousRange{0.0, 1.0}, hpoea::core::Transform::sqrt);
            }},
            {"optimize rejects continuous lower > upper", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("p", hpoea::core::ContinuousRange{10.0, 1.0});
            }},
            {"optimize rejects integer lower > upper", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("p", hpoea::core::IntegerRange{10, 1});
            }},
            {"validate rejects fixed value outside parameter range", true, [&] {
                ParameterSpace p;
                ParameterDescriptor desc;
                desc.name = "x";
                desc.type = ParameterType::Continuous;
                desc.continuous_range = hpoea::core::ContinuousRange{0.0, 1.0};
                p.add_descriptor(desc);
                hpoea::core::SearchSpace s;
                s.fix("x", 5.0);
                s.validate(p);
            }},
            {"set: rejects integer bounds lower > upper", true, [&] {
                hpoea::core::SearchSpace s;
                hpoea::core::ParameterConfig config;
                config.mode = SearchMode::optimize;
                config.integer_bounds = hpoea::core::IntegerRange{10, 1};
                s.set("pop", config);
            }},
            {"set: rejects log transform with non-positive lower bound", true, [&] {
                hpoea::core::SearchSpace s;
                hpoea::core::ParameterConfig config;
                config.mode = SearchMode::optimize;
                config.continuous_bounds = hpoea::core::ContinuousRange{0.0, 1.0};
                config.transform = hpoea::core::Transform::log;
                s.set("lr", config);
            }},
            {"validate: integer discrete_choices rejects wrong variant type", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize_choices("pop", {std::string{"wrong_type"}});
                s.validate(params);
            }},
            {"validate: integer discrete_choices rejects out-of-range value", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize_choices("pop", {std::int64_t{99}});
                s.validate(params);
            }},
            {"validate: categorical discrete_choices rejects wrong variant type", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize_choices("mode", {std::int64_t{1}});
                s.validate(params);
            }},
            {"validate: categorical discrete_choices rejects invalid value", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize_choices("mode", {std::string{"c"}});
                s.validate(params);
            }},
            {"validate: continuous bounds on integer param throws", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("pop", hpoea::core::ContinuousRange{0.0, 1.0});
                s.validate(params);
            }},
            {"validate: integer bounds on continuous param throws", true, [&] {
                hpoea::core::SearchSpace s;
                s.optimize("lr", hpoea::core::IntegerRange{0, 10});
                s.validate(params);
            }},
        };
        for (const auto &c : cases) {
            bool threw = false;
            try {
                c.fn();
            } catch (const hpoea::core::ParameterValidationError &) {
                threw = true;
            }
            HPOEA_V2_CHECK(runner, threw == c.expect_throw, c.label);
        }
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
        space.fix("lr", 0.5);
        HPOEA_V2_CHECK(runner, space.get("lr")->mode == SearchMode::fixed, "set overwrite: initially fixed");
        space.optimize("lr", hpoea::core::ContinuousRange{0.1, 0.9});
        HPOEA_V2_CHECK(runner, space.get("lr")->mode == SearchMode::optimize, "set overwrite: now optimize");
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
        HPOEA_V2_CHECK(runner, mode_eb && mode_eb->discrete_choice_count == 0u,
                       "effective_bounds: categorical descriptor without custom choices keeps descriptor choices implicit");
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
        bool found_epochs = false;
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
            } else if (eb.name == "epochs") {
                found_epochs = true;
                HPOEA_V2_CHECK(runner, eb.mode == hpoea::core::SearchMode::optimize,
                               "get_effective_bounds: epochs mode is optimize");
                HPOEA_V2_CHECK(runner, eb.integer_bounds && eb.integer_bounds->lower == 1 && eb.integer_bounds->upper == 100,
                               "get_effective_bounds: epochs integer bounds preserved");
            }
        }
        HPOEA_V2_CHECK(runner, found_lr, "get_effective_bounds: found lr entry");
        HPOEA_V2_CHECK(runner, found_epochs, "get_effective_bounds: found epochs entry");
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
